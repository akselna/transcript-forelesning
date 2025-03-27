from flask import Flask, request, jsonify, render_template, send_file
import os
import subprocess
import ffmpeg
import json
import cv2
import numpy as np
from datetime import timedelta
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import shutil
import time
import threading


transcription_progress = {
    'current_job': None,
    'total_intervals': 0,
    'current_interval': 0,
    'processed_duration': 0,
    'total_duration': 0,
    'start_time': 0,
    'estimated_completion': '',
    'time_remaining': '',
    'percent_complete': 0,
    'status': 'idle',  # 'idle', 'processing', 'completed', 'error'
    'error_message': ''
}

progress_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static/images'
SLIDES_FOLDER = 'static/slides'
WHISPER_CPP_PATH = os.environ.get('WHISPER_CPP_PATH', '/path/to/whisper.cpp')  # Configure this
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'small')  # Default model size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['SLIDES_FOLDER'] = SLIDES_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(SLIDES_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# Function to transcribe audio using whisper.cpp
def transcribe_with_whisper_cpp(audio_path, language='sv'):
    """
    Transcribe audio file using whisper.cpp
    
    Parameters:
    - audio_path: Path to the WAV audio file
    - language: Language code (default: 'no' for Norwegian)
    
    Returns:
    - Transcription text as string
    """
    try:
        # Path to whisper.cpp directory
        whisper_dir = os.environ.get('WHISPER_CPP_PATH', '/path/to/whisper.cpp')
        model_name = os.environ.get('WHISPER_MODEL', 'small')
        
        # Check all possible executable locations
        possible_executables = [
            os.path.join(whisper_dir, 'bin', 'whisper-cli'),
            os.path.join(whisper_dir, 'build', 'bin', 'whisper-cli'),
            os.path.join(whisper_dir, 'bin', 'main'),
            os.path.join(whisper_dir, 'build', 'bin', 'main'),
            os.path.join(whisper_dir, 'main')
        ]
        
        whisper_exe = None
        for exe_path in possible_executables:
            if os.path.exists(exe_path) and os.access(exe_path, os.X_OK):
                whisper_exe = exe_path
                logger.info(f"Found whisper.cpp executable at: {whisper_exe}")
                break
                
        if not whisper_exe:
            raise FileNotFoundError(f"Could not find whisper.cpp executable in {whisper_dir}")
        
        # Find the model file
        models_dir = os.path.join(whisper_dir, 'models')
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
            
        # Look for any model file matching the pattern
        model_files = []
        for file in os.listdir(models_dir):
            if file.startswith(f'ggml-{model_name}') and file.endswith('.bin'):
                model_files.append(file)
                
        if not model_files:
            raise FileNotFoundError(f"No model file found for '{model_name}' in {models_dir}")
        
        # Use the first matching model file
        model_path = os.path.join(models_dir, model_files[0])
        logger.info(f"Using model file: {model_path}")
        
        # Build command - adjust based on which executable is found
        if 'whisper-cli' in whisper_exe:
            # New CLI format (newer versions of whisper.cpp)
            cmd = [
                whisper_exe,
                '-m', model_path,
                '-f', audio_path,
                '-l', language,
                '-otxt'  # Output as text file
            ]
        else:
            # Legacy main format
            cmd = [
                whisper_exe,
                '-m', model_path,
                '-f', audio_path,
                '--language', language,
                '-otxt'  # Output as text file
            ]
        
        # Get the output directory and filename without extension
        output_dir = os.path.dirname(audio_path)
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Expected output file
        output_file = os.path.join(output_dir, f"{base_filename}.txt")
        
        # Execute the command
        logger.info(f"Executing whisper.cpp command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if there was an error
        if result.returncode != 0:
            logger.error(f"whisper.cpp error: {result.stderr}")
            # Also log stdout, which might contain useful info
            logger.error(f"whisper.cpp stdout: {result.stdout}")
            raise RuntimeError(f"whisper.cpp failed with error: {result.stderr}")
        
        # Check if output file exists
        if not os.path.exists(output_file):
            logger.warning(f"Expected output file not found: {output_file}")
            
            # Try alternative output file format (some versions use different naming)
            alt_output_file = os.path.join(output_dir, f"{base_filename}.wav.txt")
            if os.path.exists(alt_output_file):
                logger.info(f"Found alternative output file: {alt_output_file}")
                output_file = alt_output_file
            else:
                # If we still can't find the output file, try to extract from stdout
                logger.warning("Using stdout content instead")
                clean_output = result.stdout
                # Try to clean up the output - remove progress bars and other non-transcript content
                lines = clean_output.split('\n')
                # Filter out lines that look like progress information
                transcript_lines = [line for line in lines if not line.startswith(('[', 'whisper_', 'Transcription'))]
                return '\n'.join(transcript_lines).strip()
        
        # Read the transcription from the output file
        with open(output_file, 'r', encoding='utf-8') as f:
            transcription = f.read()
            
        # Clean up the output file
        try:
            os.remove(output_file)
        except Exception as e:
            logger.warning(f"Could not remove output file: {e}")
            
        return transcription
        
    except Exception as e:
        logger.error(f"Error in transcribe_with_whisper_cpp: {str(e)}", exc_info=True)
        raise
def detect_slide_changes(video_path, threshold=30, min_slide_duration=3, capture_seconds_before_change=2):
    """
    Enhanced slide detection that captures frames 2 seconds BEFORE a slide change is detected.
    
    Parameters:
    - video_path: Path to the video file
    - threshold: Threshold for detecting changes (0-100, higher = less sensitive)
    - min_slide_duration: Minimum duration a slide should appear (seconds)
    - capture_seconds_before_change: How many seconds before the change to capture (default 2)
    
    Returns:
    - List of dictionaries with detected slides: {time, end_time, image_path}
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {fps} FPS, {total_frames} frames, {width}x{height} resolution")
    
    # Create output directory for slides
    slides_dir = app.config['SLIDES_FOLDER']
    os.makedirs(slides_dir, exist_ok=True)
    
    # Calculate region of interest (central portion of the screen)
    roi_x_start = int(width * 0.15)
    roi_y_start = int(height * 0.15)
    roi_width = int(width * 0.7)
    roi_height = int(height * 0.7)
    
    # Calculate number of frames to look back
    frames_before_change = int(capture_seconds_before_change * fps)
    
    # Frame buffer to store recent frames
    frame_buffer = []
    buffer_size = frames_before_change + 10  # Add some extra buffer
    
    # Variables for tracking slide changes
    prev_frame = None
    frame_count = 0
    slide_changes = []
    
    # Convert minimum slide duration to frames, considering our sampling rate
    min_slide_duration_seconds = min_slide_duration
    
    # Keep track of the real-time position of the last change
    last_change_time = -min_slide_duration_seconds
    
    # Process frames with a sampling rate to improve performance
    if total_frames > 10000:
        sampling_rate = max(1, int(fps * 0.5))  # Sample at 0.5 frame per second for long videos
    else:
        sampling_rate = max(1, int(fps * 0.25))  # Sample at 1/4 the frame rate for shorter videos
    
    # Set a maximum number of frames to process
    max_frames_to_process = min(total_frames, 90 * 60 * fps)  # Cap at 90 minutes
    
    progress_interval = max(1, int(max_frames_to_process / 20))  # Report progress 20 times
    
    logger.info(f"Starting enhanced slide detection with sampling rate: {sampling_rate}, " 
                f"processing up to {max_frames_to_process} frames, capturing frames {capture_seconds_before_change}s before change")
    
    # IMPORTANT: Always capture the first frame as the first slide
    ret, first_frame = cap.read()
    if ret:
        first_slide_path = os.path.join(slides_dir, "slide_1.jpg")
        cv2.imwrite(first_slide_path, first_frame)
        
        first_timestamp = 0
        formatted_time = str(timedelta(seconds=int(first_timestamp)))
        
        # Add first slide to the results
        slide_changes.append({
            'frame': 0,
            'time': formatted_time,
            'time_seconds': first_timestamp,
            'image_path': first_slide_path
        })
        logger.info(f"Captured first slide at 0:00:00")
        
        # Initialize with the first frame
        prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
        
        # Initialize buffer with the first frame
        frame_buffer.append({
            'frame_count': 0,
            'timestamp': 0,
            'image': first_frame.copy()
        })
        
        # Set the last change time to the beginning
        last_change_time = 0
        
        # Move to the first frame we'll process based on sampling rate
        frame_count = sampling_rate
    else:
        logger.error("Could not read the first frame of the video")
        return []
    
    # Helper function to calculate structural similarity
    def calculate_similarity(img1, img2):
        # Extract ROI from both images
        roi1 = img1[roi_y_start:roi_y_start+roi_height, roi_x_start:roi_x_start+roi_width]
        roi2 = img2[roi_y_start:roi_y_start+roi_height, roi_x_start:roi_x_start+roi_width]
        
        # Calculate absolute difference
        frame_delta = cv2.absdiff(roi1, roi2)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Calculate percentage of changed pixels
        change_percentage = (np.count_nonzero(thresh) / thresh.size) * 100
        return change_percentage
    
    while frame_count < max_frames_to_process:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Reached end of video at frame {frame_count}/{total_frames}")
            break
            
        # Calculate current time in seconds
        current_time_seconds = frame_count / fps
        
        # Add to buffer
        frame_buffer.append({
            'frame_count': frame_count,
            'timestamp': current_time_seconds,
            'image': frame.copy()
        })
        
        # Keep buffer size limited
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)
            
        # Convert to grayscale and apply slight blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Calculate change percentage using the central portion of the screen
        change_percentage = calculate_similarity(prev_frame, gray)
        
        # If significant change detected
        adjusted_threshold = threshold / 100  # Convert threshold to percentage
        if change_percentage > adjusted_threshold:
            # Check if enough time has passed since the last change
            time_since_last_change = current_time_seconds - last_change_time
            
            if time_since_last_change >= min_slide_duration_seconds:
                # Find the frame from X seconds before the change
                capture_time = current_time_seconds - capture_seconds_before_change
                capture_frame = None
                capture_frame_idx = -1
                
                # Find the closest frame in the buffer
                for i, buf_frame in enumerate(frame_buffer):
                    if buf_frame['timestamp'] <= capture_time:
                        capture_frame = buf_frame['image']
                        capture_frame_idx = i
                
                # If we couldn't find a frame from earlier (might happen near the start),
                # just use the earliest frame in the buffer
                if capture_frame is None and frame_buffer:
                    capture_frame = frame_buffer[0]['image']
                    capture_frame_idx = 0
                
                # If we found a frame to capture
                if capture_frame is not None:
                    # Get actual timestamp of the captured frame
                    capture_timestamp = frame_buffer[capture_frame_idx]['timestamp']
                    formatted_capture_time = str(timedelta(seconds=int(capture_timestamp)))
                    
                    # If this is not the first change, update the end time of the previous slide
                    if len(slide_changes) > 0:
                        prev_slide = slide_changes[-1]
                        prev_slide['end_frame'] = frame_count
                        prev_slide['end_time'] = formatted_capture_time
                        prev_slide['end_time_seconds'] = capture_timestamp
                    
                    # Create a new slide entry
                    new_slide_path = os.path.join(slides_dir, f"slide_{len(slide_changes)+1}.jpg")
                    cv2.imwrite(new_slide_path, capture_frame)
                    
                    slide_changes.append({
                        'frame': frame_count,
                        'time': formatted_capture_time,
                        'time_seconds': capture_timestamp,
                        'image_path': new_slide_path
                    })
                    
                    logger.info(f"Detected slide change at {str(timedelta(seconds=int(current_time_seconds)))}, "
                               f"captured frame from {formatted_capture_time} "
                               f"({capture_seconds_before_change}s before change)")
                    
                    # Update the last change time
                    last_change_time = current_time_seconds
                else:
                    logger.warning(f"Could not find a frame to capture at {capture_time}s")
            else:
                logger.debug(f"Ignoring potential slide change at {current_time_seconds:.1f}s - too soon after previous change ({time_since_last_change:.1f}s < {min_slide_duration_seconds}s)")
            
        # Update previous frame
        prev_frame = gray
        frame_count += sampling_rate
        
        # Report progress periodically
        if frame_count % progress_interval == 0:
            progress = (frame_count / max_frames_to_process) * 100
            logger.info(f"Processing: {progress:.1f}% complete ({frame_count}/{max_frames_to_process} frames)")
    
    # Handle the last slide's end time
    if len(slide_changes) > 0:
        last_slide = slide_changes[-1]
        if not last_slide.get('end_time'):
            # Set the end time to the end of the video
            last_frame_position = min(total_frames - 1, max_frames_to_process - 1)
            video_duration = last_frame_position / fps
            end_formatted_time = str(timedelta(seconds=int(video_duration)))
            
            last_slide['end_frame'] = last_frame_position
            last_slide['end_time'] = end_formatted_time
            last_slide['end_time_seconds'] = video_duration
    
    # Post-processing: Remove slides that are too short
    filtered_slides = []
    for i in range(len(slide_changes)):
        slide = slide_changes[i]
        
        # Check if this slide has a valid duration
        if slide.get('end_time_seconds') and slide['time_seconds'] is not None:
            duration = slide['end_time_seconds'] - slide['time_seconds']
            # Always keep the first slide
            if i == 0 or duration >= min_slide_duration_seconds:
                filtered_slides.append(slide)
            else:
                logger.info(f"Removing slide {i+1} with duration {duration:.1f}s < {min_slide_duration_seconds}s")
                # Remove the image file
                if os.path.exists(slide['image_path']):
                    try:
                        os.remove(slide['image_path'])
                    except Exception as e:
                        logger.error(f"Error removing slide image: {str(e)}")
        else:
            # If we don't have duration info, keep the slide
            filtered_slides.append(slide)
    
    # Release video capture
    cap.release()
    
    # Renumber the slides and image files
    final_slides = []
    for i, slide in enumerate(filtered_slides):
        # Create new image path with correct numbering
        new_image_path = os.path.join(slides_dir, f"slide_{i+1}.jpg")
        
        # Rename the image file if needed
        if slide['image_path'] != new_image_path and os.path.exists(slide['image_path']):
            try:
                if os.path.exists(new_image_path):
                    os.remove(new_image_path)  # Remove any existing file with the same name
                os.rename(slide['image_path'], new_image_path)
            except Exception as e:
                logger.error(f"Error renaming slide image: {str(e)}")
                # If rename fails, copy the file
                try:
                    shutil.copy2(slide['image_path'], new_image_path)
                except Exception as e2:
                    logger.error(f"Error copying slide image: {str(e2)}")
        
        # Update the slide with the new image path
        slide['image_path'] = new_image_path
        final_slides.append(slide)
    
    logger.info(f"Slide detection completed. Found {len(final_slides)} slides after filtering.")
    return final_slides


@app.route('/transcription_progress', methods=['GET'])
def get_transcription_progress():
    """API endpoint to get the current transcription progress"""
    with progress_lock:
        return jsonify(transcription_progress)

def format_for_interface(slide_changes):
    """
    Format the detected slide changes for use in the web interface.
    """
    formatted_slides = []
    
    for slide in slide_changes:
        # Format time strings as HH:MM:SS
        start_parts = slide['time'].split(':')
        if len(start_parts) == 2:  # If format is MM:SS
            start_time = f"00:{start_parts[0]}:{start_parts[1]}"
        else:
            start_time = slide['time']
            
        end_parts = slide['end_time'].split(':')
        if len(end_parts) == 2:  # If format is MM:SS
            end_time = f"00:{end_parts[0]}:{end_parts[1]}"
        else:
            end_time = slide['end_time']
        
        formatted_slides.append({
            'start': start_time,
            'end': end_time,
            'image': '/' + slide['image_path'],  # Web path
            'image_filename': os.path.basename(slide['image_path']),
            'text': ''  # Will be filled with transcription later
        })
    
    return formatted_slides

@app.route('/detect_slides', methods=['POST'])
def detect_slides():
    if 'video' not in request.files:
        return jsonify({'error': 'Ingen video lastet opp'}), 400

    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)
    
    try:
        # Get sensitivity parameters if provided
        sensitivity = int(request.form.get('sensitivity', 30))  # 0-100, default 30
        min_duration = int(request.form.get('min_duration', 3))  # seconds, default 3
        capture_before = float(request.form.get('capture_before', 2.0))  # seconds before change to capture
        
        # Detect slide changes
        logger.info(f"Starting slide detection with sensitivity={sensitivity}, min_duration={min_duration}, capture_before={capture_before}s")
        
        slide_changes = detect_slide_changes(
            video_path, 
            threshold=sensitivity, 
            min_slide_duration=min_duration,
            capture_seconds_before_change=capture_before
        )
        
        # Check if we found any slides
        if not slide_changes:
            return jsonify({'error': 'Ingen lysbilder funnet. Prøv å justere følsomheten.'}), 400
            
        # Format for the interface
        formatted_slides = format_for_interface(slide_changes)
        
        return jsonify({
            'slides': formatted_slides,
            'video_path': video_path
        })
    
    except Exception as e:
        logger.error(f"Error detecting slides: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
import zipfile
import io
import os
from pathlib import Path

@app.route('/generate_text_pdf', methods=['POST'])
def generate_text_pdf():
    """Generate a PDF with just transcript text (no images)"""
    try:
        results = request.json
        logger.debug(f"Generating text-only PDF for {len(results)} slides")
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transcript.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        y_position = height - 50
        styles = getSampleStyleSheet()
        style = styles['Normal']
        style.fontSize = 12
        
        # Add AI guidance prompt at the beginning of the PDF (in English)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_position, "AI Guidance Instructions")
        y_position -= 30
        
        c.setFont("Helvetica", 11)
        
        ai_prompt = [
            "This document contains lecture transcriptions with references to separate slide images,",
            "organized chronologically. Each transcript entry corresponds to what was said during the",
            "timeframe when a specific slide was shown.",
            "",
            "The corresponding slide images are provided in a separate ZIP file, with filenames matching",
            "the slide numbers referenced in this document (e.g., 'slide_1.png', 'slide_2.png', etc.).",
            "",
            "As an AI assistant processing this material, please:",
            "",
            "1. Extract and summarize the key concepts and main ideas presented in this lecture",
            "2. Explain these concepts in a highly pedagogical way, as if teaching someone with no prior knowledge",
            "3. Identify relationships between different concepts and create a logical knowledge structure",
            "4. Highlight important terminology, definitions, examples, and applications",
            "5. Break down complex ideas into simple, understandable components",
            "6. Provide analogies or metaphors where appropriate to aid understanding",
            "7. Create a summary that captures the essence of the lecture in 3-5 main points",
            "8. Refer to the visual content from slides when relevant to your explanation",
            "",
            "Remember to focus on depth of understanding rather than breadth, prioritizing mastery of",
            "fundamental principles over covering every detail. Present information in a structured way that",
            "builds knowledge progressively, connecting new information to previously explained concepts.",
            "",
            "Begin your response with a concise overview of what this lecture covers and why it matters."
        ]
        
        # Draw the AI prompt text
        for line in ai_prompt:
            c.drawString(50, y_position, line)
            y_position -= 15
        
        # Add extra space after the prompt
        y_position -= 30
        
        # Add a separator line
        c.line(50, y_position, width - 50, y_position)
        y_position -= 30
        
        # Start processing slides
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_position, "Lecture Transcript")
        y_position -= 30
        
        for i, item in enumerate(results, 1):
            logger.debug(f"Processing slide {i} transcript")
            
            # Check if we need a new page
            if y_position < height/2:
                c.showPage()
                y_position = height - 50
            
            # Slide header
            c.setFont("Helvetica-Bold", 14)
            slide_ref = f"Slide {i} ({item['start']} - {item['end']})"
            c.drawString(50, y_position, slide_ref)
            y_position -= 30

            # Transcription
            c.setFont("Helvetica", 12)
            c.drawString(50, y_position, "Transcript:")
            y_position -= 20

            # Handle text with proper wrapping manually
            try:
                # Safely handle the text content
                text = item.get('text', '')
                if not text:
                    text = "No text available"
                
                # Replace newlines with HTML breaks
                text = text.replace('\n', '<br />')
                
                # Simple text wrapping approach
                available_width = width - 100
                words = text.split()
                lines = []
                line = ""
                
                for word in words:
                    # Handle HTML breaks
                    if '<br />' in word:
                        parts = word.split('<br />')
                        for j, part in enumerate(parts):
                            if j > 0:
                                # Add the current line and start a new one
                                if line:
                                    lines.append(line)
                                line = part
                            else:
                                if line:
                                    test_line = line + " " + part
                                    if c.stringWidth(test_line, "Helvetica", 11) < available_width:
                                        line = test_line
                                    else:
                                        lines.append(line)
                                        line = part
                                else:
                                    line = part
                    else:
                        # Normal word, check if it fits on the current line
                        if line:
                            test_line = line + " " + word
                            if c.stringWidth(test_line, "Helvetica", 11) < available_width:
                                line = test_line
                            else:
                                lines.append(line)
                                line = word
                        else:
                            line = word
                
                # Add the last line if there is one
                if line:
                    lines.append(line)
                
                # Calculate the height needed for all lines
                line_height = 15  # Approximate height per line
                text_height = len(lines) * line_height
                
                # Check if we need a new page
                if y_position - text_height < 50:
                    logger.debug(f"Creating new page for slide {i} text")
                    c.showPage()
                    y_position = height - 50
                    
                    # Repeat the header on the new page
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(50, y_position, f"{slide_ref} (continued)")
                    y_position -= 30
                    
                    c.setFont("Helvetica", 12)
                    c.drawString(50, y_position, "Transcript:")
                    y_position -= 20
                
                # Draw each line of text
                c.setFont("Helvetica", 11)
                for line in lines:
                    c.drawString(50, y_position, line)
                    y_position -= line_height
                
                # Add some extra space after the text
                y_position -= 20
                
            except Exception as e:
                logger.error(f"Error processing text for slide {i}: {str(e)}", exc_info=True)
                c.drawString(50, y_position, f"Error processing text: {str(e)}")
                y_position -= 40

            # Add extra space between slides
            y_position -= 20
            if y_position < 50:
                c.showPage()
                y_position = height - 50

        # Save the PDF
        c.save()
        logger.debug(f"Text-only PDF saved to: {pdf_path}")
        
        # Send the file
        return send_file(pdf_path, as_attachment=True, download_name="transcript.pdf")
        
    except Exception as e:
        logger.error(f"Error generating text PDF: {str(e)}", exc_info=True)
        return jsonify({'error': f"Could not generate text PDF: {str(e)}"}), 500

@app.route('/generate_images_zip', methods=['POST'])
def generate_images_zip():
    """Generate a ZIP file containing all slide images"""
    try:
        results = request.json
        logger.debug(f"Generating ZIP with {len(results)} slide images")
        
        # Create a BytesIO object to store the ZIP file
        memory_file = io.BytesIO()
        
        # Create a ZIP file in memory
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, item in enumerate(results, 1):
                image_path = item.get('image')
                if image_path:
                    # Convert web path to file system path
                    if image_path.startswith('/'):
                        image_path = image_path[1:]  # Remove leading slash
                    
                    # Create absolute path
                    abs_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
                    
                    if os.path.exists(abs_image_path):
                        # Determine image extension
                        img_ext = Path(abs_image_path).suffix  # Get the file extension
                        if not img_ext:
                            img_ext = '.jpg'  # Default to jpg if no extension
                        
                        # Add the image to the ZIP file with a standardized name
                        zf.write(abs_image_path, f"slide_{i}{img_ext}")
                        logger.debug(f"Added {abs_image_path} as slide_{i}{img_ext} to ZIP")
                    else:
                        logger.warning(f"Image file not found: {abs_image_path}")
        
        # Seek to the beginning of the BytesIO object
        memory_file.seek(0)
        
        # Return the ZIP file for download
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='slides.zip'
        )
    
    except Exception as e:
        logger.error(f"Error generating ZIP: {str(e)}", exc_info=True)
        return jsonify({'error': f"Could not generate ZIP file: {str(e)}"}), 500

@app.route('/download_separate', methods=['POST'])
def download_separate():
    """Handler for downloading text and images separately"""
    try:
        results = request.json
        # Track which downloads are complete
        downloads = {
            'pdf': {'url': '/generate_text_pdf', 'complete': False},
            'zip': {'url': '/generate_images_zip', 'complete': False}
        }
        
        return jsonify({'message': 'Files ready for download', 'downloads': downloads})
    except Exception as e:
        logger.error(f"Error preparing separate downloads: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    global transcription_progress
    
    if 'video' not in request.files and 'video_path' not in request.form:
        return jsonify({'error': 'Ingen video angitt'}), 400

    # Get video path - either from a new upload or from previous processing
    if 'video' in request.files:
        video_file = request.files['video']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
    else:
        video_path = request.form['video_path']
        if not os.path.exists(video_path):
            return jsonify({'error': 'Videofilen finnes ikke'}), 400

    intervals = json.loads(request.form.get('intervals', '[]'))
    if not intervals:
        return jsonify({'error': 'Ingen intervaller definert'}), 400

    try:
        # Reset progress state
        with progress_lock:
            job_id = str(int(time.time()))  # Use timestamp as job ID
            transcription_progress.update({
                'current_job': job_id,
                'total_intervals': len(intervals),
                'current_interval': 0,
                'processed_duration': 0,
                'total_duration': 0,
                'start_time': time.time(),
                'estimated_completion': '',
                'time_remaining': '',
                'percent_complete': 0,
                'status': 'processing',
                'error_message': ''
            })
        
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, format='wav', acodec='pcm_s16le', ar='16k')
        ffmpeg.run(stream, overwrite_output=True)

        # Calculate total duration of all clips for progress tracking
        total_duration = 0
        for interval in intervals:
            start = time_to_seconds(interval['start'])
            end = time_to_seconds(interval['end'])
            if start < end:
                total_duration += (end - start)
        
        # Update total duration in progress state
        with progress_lock:
            transcription_progress['total_duration'] = total_duration
        
        # Estimate time per second of audio (whisper.cpp is faster!)
        seconds_per_audio_second = 0.5  # Estimated time for processing 1 second of audio
        
        # Calculate estimated total processing time
        estimated_total_seconds = total_duration * seconds_per_audio_second
        estimated_completion = str(timedelta(seconds=int(estimated_total_seconds)))
        
        with progress_lock:
            transcription_progress['estimated_completion'] = estimated_completion
        
        logger.info(f"Starting transcription of {len(intervals)} intervals with total duration of {total_duration:.1f} seconds")
        logger.info(f"Estimated completion time: {estimated_completion}")
        
        start_time = time.time()
        processed_duration = 0
        results = []
        
        for i, interval in enumerate(intervals):
            # Update current interval in progress state
            with progress_lock:
                transcription_progress['current_interval'] = i + 1
            
            # Track progress
            segment_start = time.time()
            
            start = time_to_seconds(interval['start'])
            end = time_to_seconds(interval['end'])
            if start >= end:
                continue
            
            interval_duration = end - start
            processed_percent = (processed_duration / total_duration) * 100 if total_duration > 0 else 0
            
            # Update progress state
            with progress_lock:
                transcription_progress['processed_duration'] = processed_duration
                transcription_progress['percent_complete'] = processed_percent

            # Legg til denne kodebiten her
            if i % 5 == 0:  # Logger bare for hvert 5. intervall
                logger.info(f"Transcription progress: {processed_percent:.1f}% complete ({i}/{len(intervals)} intervals)")
        
            # Print progress information
            logger.info(f"Processing interval {i+1}/{len(intervals)}: {interval['start']} to {interval['end']} ({interval_duration:.1f}s)")
            logger.info(f"Progress: {processed_percent:.1f}% complete, {processed_duration:.1f}/{total_duration:.1f} seconds processed")
            
            # Estimate remaining time
            elapsed = time.time() - start_time
            if processed_duration > 0:
                seconds_per_audio_second_actual = elapsed / processed_duration
                remaining_duration = total_duration - processed_duration
                estimated_remaining = remaining_duration * seconds_per_audio_second_actual
                time_remaining = str(timedelta(seconds=int(estimated_remaining)))
                
                # Update time remaining in progress state
                with progress_lock:
                    transcription_progress['time_remaining'] = time_remaining
                
                logger.info(f"Estimated time remaining: {time_remaining}")
            
            # Process the clip
            clipped_audio = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_clip_{i}.wav')
            stream = ffmpeg.input(audio_path, ss=start, t=end - start)
            stream = ffmpeg.output(stream, clipped_audio, format='wav', acodec='pcm_s16le')
            ffmpeg.run(stream, overwrite_output=True)

            # Use whisper.cpp instead of Python whisper
            try:
                text = transcribe_with_whisper_cpp(clipped_audio, language='sv')
            except Exception as e:
                logger.error(f"Error using whisper.cpp: {str(e)}")
                text = f"Error: {str(e)}"

            # Get image - either from the form or from the interval data
            image_url = interval.get('image')  # For automatic detection
            image_filename = interval.get('image_filename')
            
            # If no image is provided in the interval, check if there's an image file
            if not image_url:
                image_key = f'image_{i}'
                if image_key in request.files:
                    image = request.files[image_key]
                    image_filename = image.filename
                    image_path = os.path.join(app.config['IMAGE_FOLDER'], f'slide_{i}_{image.filename}')
                    abs_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
                    image.save(abs_image_path)
                    image_url = f"/{image_path}"

            results.append({
                'start': interval['start'],
                'end': interval['end'],
                'text': text,
                'image': image_url,
                'image_filename': image_filename
            })
            
            os.remove(clipped_audio)
            
            # Update processed duration
            processed_duration += interval_duration
            
            # Calculate actual processing time for this segment
            segment_time = time.time() - segment_start
            logger.info(f"Interval {i+1} processed in {segment_time:.1f} seconds ({interval_duration/segment_time:.1f}x real-time)")

        # Calculate total processing time
        total_time = time.time() - start_time
        logger.info(f"Transcription completed in {str(timedelta(seconds=int(total_time)))}")
        logger.info(f"Average processing speed: {total_duration/total_time:.1f}x real-time")

        # Update progress state to completed
        with progress_lock:
            transcription_progress.update({
                'status': 'completed',
                'percent_complete': 100,
                'processed_duration': total_duration,
                'time_remaining': '0:00:00'
            })

        # Don't delete the video if we've just done slide detection (might need for repeated processing)
        if 'keep_video' not in request.form or request.form['keep_video'] != 'true':
            os.remove(video_path)
            
        os.remove(audio_path)
        return jsonify(results)
    except Exception as e:
        # Update progress state to error
        with progress_lock:
            transcription_progress.update({
                'status': 'error',
                'error_message': str(e)
            })
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 5000

@app.route('/generate_pdf', methods=['POST'])
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        results = request.json
        logger.debug(f"Received data for PDF generation: {json.dumps(results, indent=2)}")
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'forelesning.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        y_position = height - 50
        styles = getSampleStyleSheet()
        style = styles['Normal']
        style.fontSize = 12
        
        # Add AI guidance prompt at the beginning of the PDF
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_position, "AI Guidance Instructions")
        y_position -= 30
        
        c.setFont("Helvetica", 11)
        
        ai_prompt = [
            "This document contains lecture slides with their corresponding transcriptions, organized chronologically.",
            "Each slide includes a visual capture from the presentation and a transcript of what was said during that",
            "timeframe. The slides often contain annotations made by the lecturer during the presentation.",
            "",
            "As an AI assistant processing this material, please:",
            "",
            "1. Extract and summarize the key concepts and main ideas presented in this lecture",
            "2. Explain these concepts in a highly pedagogical way, as if teaching someone with no prior knowledge",
            "3. Identify relationships between different concepts and create a logical knowledge structure",
            "4. Highlight important terminology, definitions, examples, and applications",
            "5. Break down complex ideas into simple, understandable components",
            "6. Provide analogies or metaphors where appropriate to aid understanding",
            "7. Create a summary that captures the essence of the lecture in 3-5 main points",
            "8. If any mathematical formulas, code samples, or specialized notation appears in the slides,",
            "   explain them thoroughly and connect them to practical applications",
            "",
            "Remember to focus on depth of understanding rather than breadth, prioritizing mastery of",
            "fundamental principles over covering every detail. Present information in a structured way that",
            "builds knowledge progressively, connecting new information to previously explained concepts.",
            "",
            "Begin your response with a concise overview of what this lecture covers and why it matters."
        ]
        
        # Draw the AI prompt text
        for line in ai_prompt:
            c.drawString(50, y_position, line)
            y_position -= 15
        
        # Add extra space after the prompt
        y_position -= 30
        
        # Add a separator line
        c.line(50, y_position, width - 50, y_position)
        y_position -= 30
        
        # Start processing slides
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_position, "Lecture Content")
        y_position -= 30
        
        for i, item in enumerate(results, 1):
            logger.debug(f"Processing slide {i}")
            
            # Check if we need a new page
            if y_position < height/2:
                c.showPage()
                y_position = height - 50
            
            # Slide header
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, f"Lysbilde {i} ({item['start']} - {item['end']})")
            y_position -= 30

            # Handle image
            if item.get('image'):
                try:
                    # Convert web path to file system path
                    image_path = item['image']
                    if image_path.startswith('/'):
                        image_path = image_path[1:]  # Remove leading slash
                    
                    # Create absolute path
                    abs_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
                    logger.debug(f"Loading image from: {abs_image_path}")
                    
                    if os.path.exists(abs_image_path):
                        # Get image dimensions to preserve aspect ratio
                        img = ImageReader(abs_image_path)
                        img_width, img_height = img.getSize()
                        
                        # Calculate scaled dimensions to fit the page while preserving aspect ratio
                        max_width = width - 100  # Leave margins
                        max_height = 200  # Maximum height for images
                        
                        # Scale to fit within bounds while preserving aspect ratio
                        scale = min(max_width / img_width, max_height / img_height)
                        display_width = img_width * scale
                        display_height = img_height * scale
                        
                        # Center the image horizontally
                        x_position = (width - display_width) / 2
                        
                        # Draw the image with proper dimensions
                        c.drawImage(img, x_position, y_position - display_height, 
                                   width=display_width, height=display_height)
                        y_position -= (display_height + 20)  # Add some padding
                    else:
                        error_msg = f"Image file not found: {abs_image_path}"
                        logger.error(error_msg)
                        c.drawString(50, y_position, f"Kunne ikke laste bilde: File not found")
                        y_position -= 20
                except Exception as e:
                    logger.error(f"Error loading image: {str(e)}", exc_info=True)
                    c.drawString(50, y_position, f"Kunne ikke laste bilde: {str(e)}")
                    y_position -= 20

            # Transcription label
            c.setFont("Helvetica", 12)
            c.drawString(50, y_position, "Transkripsjon:")
            y_position -= 20

            # Handle text with proper wrapping manually
            try:
                # Safely handle the text content
                text = item.get('text', '')
                if not text:
                    text = "Ingen tekst tilgjengelig"
                
                # Replace newlines with HTML breaks
                text = text.replace('\n', '<br />')
                
                # Simple text wrapping approach
                available_width = width - 100
                words = text.split()
                lines = []
                line = ""
                
                for word in words:
                    # Handle HTML breaks
                    if '<br />' in word:
                        parts = word.split('<br />')
                        for j, part in enumerate(parts):
                            if j > 0:
                                # Add the current line and start a new one
                                if line:
                                    lines.append(line)
                                line = part
                            else:
                                if line:
                                    test_line = line + " " + part
                                    if c.stringWidth(test_line, "Helvetica", 11) < available_width:
                                        line = test_line
                                    else:
                                        lines.append(line)
                                        line = part
                                else:
                                    line = part
                    else:
                        # Normal word, check if it fits on the current line
                        if line:
                            test_line = line + " " + word
                            if c.stringWidth(test_line, "Helvetica", 11) < available_width:
                                line = test_line
                            else:
                                lines.append(line)
                                line = word
                        else:
                            line = word
                
                # Add the last line if there is one
                if line:
                    lines.append(line)
                
                # Calculate the height needed for all lines
                line_height = 15  # Approximate height per line
                text_height = len(lines) * line_height
                
                # Check if we need a new page
                if y_position - text_height < 50:
                    logger.debug(f"Creating new page for slide {i} text")
                    c.showPage()
                    y_position = height - 50
                    
                    # Repeat the header on the new page
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(50, y_position, f"Lysbilde {i} ({item['start']} - {item['end']}) (fortsetter)")
                    y_position -= 30
                    
                    c.setFont("Helvetica", 12)
                    c.drawString(50, y_position, "Transkripsjon:")
                    y_position -= 20
                
                # Draw each line of text
                c.setFont("Helvetica", 11)
                for line in lines:
                    c.drawString(50, y_position, line)
                    y_position -= line_height
                
                # Add some extra space after the text
                y_position -= 20
                
            except Exception as e:
                logger.error(f"Error processing text for slide {i}: {str(e)}", exc_info=True)
                c.drawString(50, y_position, f"Feil ved behandling av tekst: {str(e)}")
                y_position -= 40

            # Add extra space between slides
            y_position -= 20
            if y_position < 50:
                c.showPage()
                y_position = height - 50

        # Save the PDF
        c.save()
        logger.debug(f"PDF saved to: {pdf_path}")
        
        # Send the file
        return send_file(pdf_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        return jsonify({'error': f"Kunne ikke generere PDF: {str(e)}"}), 500

def time_to_seconds(time_str):
    """
    Convert a time string to seconds.
    Handles both 'HH:MM:SS' and 'MM:SS' formats.
    """
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS format
        m, s = map(int, parts)
        return m * 60 + s
    elif len(parts) == 3:  # HH:MM:SS format
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    else:
        raise ValueError(f"Invalid time format: {time_str}")

if __name__ == '__main__':
    app.run(debug=True)