Brain Tumor Report Generator

Web app to upload MRI (JPG), run segmentation + tumor/plane classifiers, and download a PDF report.
Results are saved to MySQL. Optional Cloudinary for image URLs.

⸻

Features
	•	Upload MRI scans via a simple UI/UX.
	•	Tumor Segmentation using U-Net.
	•	Tumor Classification (Glioma, Meningioma, Pituitary, No Tumor).
	•	Plane Axis Detection (Axial, Coronal, Sagittal).
	•	Generate and download PDF reports.
	•	Store predictions and patient details in MySQL database.

⸻

Tech Stack
	•	Frontend: HTML, CSS, JavaScript
	•	Backend: Flask (Python)
	•	ML Models: TensorFlow/Keras (U-Net, classifier, plane axis)
	•	Database: MySQL
	•	PDF: ReportLab
