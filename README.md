# Brain Tumor Report Generator 

Web app to upload MRI (JPG), run segmentation + tumor/plane classifiers, and download a PDF report. 
Results are saved to MySQL. Optional Cloudinary for image URLs.

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **ML Models:** TensorFlow/Keras (U-Net, classifier, plane axis)  
- **Database:** MySQL  
- **PDF:** ReportLab  

## Run Locally
```bash
cd backend
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
