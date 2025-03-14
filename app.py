from flask import Flask, request, jsonify, render_template, send_file
import os
import whisper
import ffmpeg
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'Ingen video lastet opp'}), 400

    video_file = request.files['video']
    intervals = json.loads(request.form.get('intervals', '[]'))
    if not intervals:
        return jsonify({'error': 'Ingen intervaller definert'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    try:
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, format='wav', acodec='pcm_s16le', ar='16k')
        ffmpeg.run(stream, overwrite_output=True)

        model = whisper.load_model('medium')
        results = []
        for i, interval in enumerate(intervals):
            start = time_to_seconds(interval['start'])
            end = time_to_seconds(interval['end'])
            if start >= end:
                continue

            clipped_audio = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_clip_{i}.wav')
            stream = ffmpeg.input(audio_path, ss=start, t=end - start)
            stream = ffmpeg.output(stream, clipped_audio, format='wav', acodec='pcm_s16le')
            ffmpeg.run(stream, overwrite_output=True)

            result = model.transcribe(clipped_audio, language='no')
            text = result['text']

            image_key = f'image_{i}'
            image_url = None
            image_filename = None
            if image_key in request.files:
                image = request.files[image_key]
                # Store the original filename for the PDF generator
                image_filename = image.filename
                # Create a web-accessible path for frontend display
                image_path = os.path.join(app.config['IMAGE_FOLDER'], f'slide_{i}_{image.filename}')
                # Create an absolute path for the file system
                abs_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
                image.save(abs_image_path)
                # Web path for frontend display
                image_url = f"/{image_path}"

            results.append({
                'start': interval['start'],
                'end': interval['end'],
                'text': text,
                'image': image_url,
                'image_filename': image_filename
            })
            os.remove(clipped_audio)

        os.remove(video_path)
        os.remove(audio_path)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
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
        
        for i, item in enumerate(results, 1):
            logger.debug(f"Processing slide {i}")
            
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
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

if __name__ == '__main__':
    app.run(debug=True)