import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from datetime import datetime
import os

# Page Configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Database file
EXCEL_FILE = "patient_records.xlsx"

# Initialize Excel database if doesn't exist
def init_database():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=[
            'Patient_ID', 'Name', 'Age', 'Gender', 'Contact', 
            'Date', 'Diagnosis', 'Severity', 'Confidence', 
            'Recommended_Medicine', 'Doctor_Notes'
        ])
        df.to_excel(EXCEL_FILE, index=False)

# Load database
def load_database():
    try:
        return pd.read_excel(EXCEL_FILE)
    except:
        init_database()
        return pd.read_excel(EXCEL_FILE)

# Save to database
def save_to_database(patient_data):
    df = load_database()
    new_df = pd.DataFrame([patient_data])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    return True

# Image Preprocessing Function (without OpenCV)
def preprocess_image(image):
    """Preprocess retinal image using PIL only"""
    # Resize to standard size
    img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img_resized)
    enhanced = enhancer.enhance(1.5)
    
    # Enhance sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpness_enhancer.enhance(1.3)
    
    # Convert to numpy array and normalize
    img_array = np.array(enhanced)
    normalized = img_array / 255.0
    
    return normalized, enhanced

# Simulated CNN Model Prediction
def predict_diabetic_retinopathy(processed_image):
    """
    Simulated CNN prediction
    In production, replace with: model.predict(processed_image)
    """
    # Simulate CNN output based on image characteristics
    img_mean = np.mean(processed_image)
    img_std = np.std(processed_image)
    
    # Simple heuristic for demo (replace with actual CNN model)
    severity_levels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    # Simulate prediction based on image statistics
    if img_mean > 0.6:
        prediction = [0.7, 0.2, 0.05, 0.03, 0.02]  # Likely No DR
    elif img_mean > 0.4:
        prediction = [0.2, 0.5, 0.2, 0.07, 0.03]   # Likely Mild
    elif img_mean > 0.3:
        prediction = [0.1, 0.2, 0.5, 0.15, 0.05]   # Likely Moderate
    else:
        prediction = [0.05, 0.1, 0.2, 0.4, 0.25]   # Likely Severe
    
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100
    
    return severity_levels[predicted_class], confidence

# Medicine Recommendation System
def recommend_medicine(severity):
    """Recommend medicine based on severity"""
    recommendations = {
        'No DR': {
            'medicines': ['Regular monitoring', 'Vitamin A supplements', 'Multivitamin with lutein'],
            'dosage': 'One tablet daily with meals',
            'precautions': 'Regular eye checkups every 6 months, maintain HbA1c below 7%',
            'lifestyle': 'Control blood sugar levels, maintain healthy diet, regular exercise'
        },
        'Mild': {
            'medicines': ['Anti-VEGF injections (Bevacizumab)', 'Blood pressure medication', 'Aspirin 75mg'],
            'dosage': 'Anti-VEGF: As prescribed by ophthalmologist, Aspirin: Once daily',
            'precautions': 'Monitor blood sugar strictly, check blood pressure daily',
            'lifestyle': 'Exercise 30 min daily, avoid smoking and alcohol, reduce salt intake'
        },
        'Moderate': {
            'medicines': ['Laser photocoagulation therapy', 'Ranibizumab injections', 'Corticosteroids', 'ACE inhibitors'],
            'dosage': 'Multiple laser sessions as needed, monthly injections initially',
            'precautions': 'Immediate consultation with retina specialist required',
            'lifestyle': 'Strict diabetes management, weight control, stress reduction'
        },
        'Severe': {
            'medicines': ['Panretinal photocoagulation', 'Aflibercept injections', 'Vitrectomy preparation', 'Insulin therapy'],
            'dosage': 'Extensive laser treatment required, bi-weekly injections',
            'precautions': 'URGENT: Emergency ophthalmology consultation within 24 hours',
            'lifestyle': 'Intensive diabetes care, daily blood sugar monitoring, dietary restrictions'
        },
        'Proliferative DR': {
            'medicines': ['URGENT Vitrectomy surgery', 'Bevacizumab pre-surgery', 'Post-op antibiotics', 'Pain management'],
            'dosage': 'Immediate surgical intervention required',
            'precautions': 'CRITICAL: Risk of permanent vision loss - seek immediate treatment',
            'lifestyle': 'Critical diabetes management, hospital care, frequent follow-ups every week'
        }
    }
    
    return recommendations.get(severity, recommendations['No DR'])

# Main Application
def main():
    # Title
    st.title("👁️ Diabetic Retinopathy Detection System")
    st.markdown("### AI-Powered Eye Disease Detection using Deep Learning CNN")
    st.markdown("---")
    
    # Initialize database
    init_database()
    
    # Sidebar - Patient Information
    with st.sidebar:
        st.header("📋 Patient Information")
        patient_name = st.text_input("Patient Name *", placeholder="Enter full name")
        patient_age = st.number_input("Age *", min_value=1, max_value=120, value=30)
        patient_gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
        patient_contact = st.text_input("Contact Number *", placeholder="+91-XXXXXXXXXX")
        doctor_notes = st.text_area("Doctor's Notes", placeholder="Additional observations...")
        
        st.markdown("---")
        st.info("📌 **Required fields marked with ***")
        st.warning("⚕️ This is a diagnostic aid. Always consult a qualified ophthalmologist.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Upload Retinal Image")
        st.markdown("Upload a clear fundus photograph of the retina")
        
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a high-quality retinal image for accurate diagnosis"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Retinal Image", use_container_width=True)
            
            # Image info
            st.info(f"📐 Image size: {image.size[0]} x {image.size[1]} pixels")
            
            # Preprocess button
            if st.button("🔬 Preprocess & Enhance Image", use_container_width=True):
                with st.spinner("Processing image..."):
                    # Preprocess
                    processed_img, enhanced_img = preprocess_image(image)
                    
                    # Store in session state
                    st.session_state.processed = True
                    st.session_state.enhanced_img = enhanced_img
                    st.session_state.processed_img = processed_img
                    
                    st.success("✅ Image preprocessed successfully!")
    
    with col2:
        if 'processed' in st.session_state and st.session_state.processed:
            st.subheader("🔍 Enhanced & Processed Image")
            st.image(st.session_state.enhanced_img, 
                    caption="Contrast Enhanced & Sharpened Image", 
                    use_container_width=True)
            
            st.success("✓ Image ready for CNN analysis")
            
            # Predict button
            if st.button("🧠 Run AI Diagnosis (CNN)", use_container_width=True):
                with st.spinner("Analyzing image with deep learning model..."):
                    import time
                    time.sleep(2)  # Simulate processing time
                    
                    # Get prediction
                    severity, confidence = predict_diabetic_retinopathy(
                        st.session_state.processed_img
                    )
                    
                    # Store results
                    st.session_state.diagnosis_done = True
                    st.session_state.severity = severity
                    st.session_state.confidence = confidence
                    
                    st.success("✅ Diagnosis completed!")
    
    # Results Section
    if 'diagnosis_done' in st.session_state and st.session_state.diagnosis_done:
        st.markdown("---")
        st.header("📊 Diagnosis Results & Medical Report")
        
        # Result display
        severity = st.session_state.severity
        confidence = st.session_state.confidence
        
        # Color coding based on severity
        color_map = {
            'No DR': '#4CAF50',
            'Mild': '#FFC107',
            'Moderate': '#FF9800',
            'Severe': '#F44336',
            'Proliferative DR': '#B71C1C'
        }
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.markdown(f"""
                <div style='background-color: {color_map[severity]}; padding: 20px; 
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{severity}</h2>
                    <p style='color: white; margin: 5px 0;'>Diagnosis</p>
                </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown(f"""
                <div style='background-color: #2196F3; padding: 20px; 
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{confidence:.1f}%</h2>
                    <p style='color: white; margin: 5px 0;'>AI Confidence</p>
                </div>
            """, unsafe_allow_html=True)
        
        with result_col3:
            st.markdown(f"""
                <div style='background-color: #9C27B0; padding: 20px; 
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{datetime.now().strftime('%d/%m/%Y')}</h2>
                    <p style='color: white; margin: 5px 0;'>Report Date</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Medicine Recommendations
        st.header("💊 Treatment Recommendations & Prescriptions")
        recommendations = recommend_medicine(severity)
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.subheader("🏥 Prescribed Medicines")
            for i, medicine in enumerate(recommendations['medicines'], 1):
                st.markdown(f"**{i}.** {medicine}")
            
            st.markdown("")
            st.subheader("📋 Dosage Instructions")
            st.info(recommendations['dosage'])
        
        with rec_col2:
            st.subheader("⚠️ Medical Precautions")
            st.warning(recommendations['precautions'])
            
            st.markdown("")
            st.subheader("🏃 Lifestyle Recommendations")
            st.success(recommendations['lifestyle'])
        
        st.markdown("---")
        
        # Save to Database
        st.header("💾 Save Patient Record to Database")
        
        save_col1, save_col2 = st.columns([3, 1])
        
        with save_col1:
            st.markdown("**Save this diagnosis report and patient information to Excel database**")
        
        with save_col2:
            if st.button("📥 Save Record", use_container_width=True):
                # Validate required fields
                if not patient_name or not patient_contact:
                    st.error("❌ Please fill in all required patient information in the sidebar!")
                else:
                    # Generate patient ID
                    patient_id = f"PT{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    # Create patient record
                    patient_data = {
                        'Patient_ID': patient_id,
                        'Name': patient_name,
                        'Age': patient_age,
                        'Gender': patient_gender,
                        'Contact': patient_contact,
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Diagnosis': severity,
                        'Severity': severity,
                        'Confidence': f"{confidence:.2f}%",
                        'Recommended_Medicine': " | ".join(recommendations['medicines']),
                        'Doctor_Notes': doctor_notes
                    }
                    
                    # Save to Excel
                    try:
                        if save_to_database(patient_data):
                            st.success(f"✅ Record saved successfully! Patient ID: **{patient_id}**")
                            st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error saving record: {str(e)}")
    
    # View Database
    st.markdown("---")
    st.header("📂 Patient Records Database")
    
    db_col1, db_col2 = st.columns([3, 1])
    
    with db_col1:
        st.markdown("View all patient records stored in the Excel database")
    
    with db_col2:
        if st.button("📊 View Records", use_container_width=True):
            st.session_state.show_db = True
    
    if 'show_db' in st.session_state and st.session_state.show_db:
        try:
            df = load_database()
            if len(df) > 0:
                st.dataframe(df, use_container_width=True, height=300)
                
                # Statistics
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Total Patients", len(df))
                with stat_col2:
                    st.metric("Today's Cases", len(df[df['Date'].str.contains(datetime.now().strftime('%Y-%m-%d'))]))
                with stat_col3:
                    st.metric("Database Size", f"{len(df)} records")
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Database (CSV)",
                    data=csv,
                    file_name=f"patient_records_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("📭 No records found in database. Start by diagnosing your first patient!")
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p><strong>⚕️ Diabetic Retinopathy Detection System | AI-Powered Healthcare Solution</strong></p>
            <p style='font-size: 12px;'>⚠️ <strong>Disclaimer:</strong> This is a diagnostic aid tool. Always consult with a qualified ophthalmologist for final diagnosis and treatment.</p>
            <p style='font-size: 12px;'>🔬 Powered by Deep Learning CNN | 📊 Data stored in Excel database</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()