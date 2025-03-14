from flask import Flask, request, jsonify, render_template, send_file
import os
import whisper
import ffmpeg
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

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
            if image_key in request.files:
                image = request.files[image_key]
                image_path = os.path.join(app.config['IMAGE_FOLDER'], f'slide_{i}_{image.filename}')
                image.save(image_path)
                image_url = image_path  # Bruk lokal filsti for PDF

            results.append({
                'start': interval['start'],
                'end': interval['end'],
                'text': text,
                'image': image_url
            })
            os.remove(clipped_audio)

        os.remove(video_path)
        os.remove(audio_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    results = request.json
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'forelesning.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y_position = height - 50  # Start øverst på siden
    styles = getSampleStyleSheet()
    style = styles['Normal']
    style.fontSize = 12

    for i, item in enumerate(results, 1):
        # Overskrift: Lysbilde X (start - slutt)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, f"Lysbilde {i} ({item['start']} - {item['end']})")
        y_position -= 30

        # Bilde
        if item.get('image'):
            try:
                img = ImageReader(item['image'])
                img_width, img_height = 500, 150  # Fast størrelse for bildet
                c.drawImage(img, 50, y_position - img_height, width=img_width, height=img_height)
                y_position -= (img_height + 10)
            except Exception as e:
                c.drawString(50, y_position, f"Kunne ikke laste bilde: {str(e)}")
                y_position -= 20

        # Transkripsjon
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, "Transkripsjon:")
        y_position -= 20

        # Bruk Paragraph for å håndtere tekstbryting
        text = item['text'].replace('\n', '<br />')  # Konverter linjeskift til HTML-bryt
        p = Paragraph(text, style)
        p.wrapOn(c, width - 100, height)  # Begrens bredden til å passe på siden
        text_height = p.getBounds()[3] - p.getBounds()[1]  # Hent høyden på teksten
        space_needed = text_height + 40  # Legg til litt ekstra plass

        # Sjekk om det er nok plass på siden
        if y_position - space_needed < 50:
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, f"Lysbilde {i} ({item['start']} - {item['end']})")
            y_position -= 30
            if item.get('image'):
                c.drawImage(img, 50, y_position - img_height, width=img_width, height=img_height)
                y_position -= (img_height + 10)
            c.setFont("Helvetica", 12)
            c.drawString(50, y_position, "Transkripsjon:")
            y_position -= 20

        # Tegn teksten
        p.drawOn(c, 50, y_position - text_height)
        y_position -= (text_height + 40)  # Flytt ned etter teksten

        # Legg til litt ekstra mellomrom mellom lysbilder
        y_position -= 20
        if y_position < 50:
            c.showPage()
            y_position = height - 50

    c.save()
    return send_file(pdf_path, as_attachment=True)
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

if __name__ == '__main__':
    app.run(debug=True)