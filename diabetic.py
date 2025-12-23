import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
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

# Image Preprocessing Function
def preprocess_image(image):
    """Preprocess retinal image for CNN model"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Resize to standard size
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(img_resized.shape) == 3:
        lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    else:
        enhanced = img_resized
    
    # Normalize
    normalized = enhanced / 255.0
    
    return normalized, enhanced

# Simulated CNN Model Prediction (Replace with actual trained model)
def predict_diabetic_retinopathy(processed_image):
    """
    Simulated CNN prediction
    In production, replace with: model.predict(processed_image)
    """
    # Simulate CNN output
    severity_levels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    # Simulate prediction (random for demo)
    prediction = np.random.rand(5)
    prediction = prediction / prediction.sum()
    
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100
    
    return severity_levels[predicted_class], confidence

# Medicine Recommendation System
def recommend_medicine(severity):
    """Recommend medicine based on severity"""
    recommendations = {
        'No DR': {
            'medicines': ['Regular monitoring', 'Vitamin A supplements'],
            'dosage': 'One tablet daily',
            'precautions': 'Regular eye checkups every 6 months',
            'lifestyle': 'Control blood sugar, maintain healthy diet'
        },
        'Mild': {
            'medicines': ['Anti-VEGF injections', 'Blood pressure medication'],
            'dosage': 'As prescribed by ophthalmologist',
            'precautions': 'Monitor blood sugar strictly',
            'lifestyle': 'Exercise regularly, avoid smoking'
        },
        'Moderate': {
            'medicines': ['Laser photocoagulation therapy', 'Anti-VEGF injections', 'Corticosteroids'],
            'dosage': 'Multiple sessions as needed',
            'precautions': 'Immediate consultation with retina specialist',
            'lifestyle': 'Strict diabetes management required'
        },
        'Severe': {
            'medicines': ['Panretinal photocoagulation', 'Anti-VEGF therapy', 'Vitrectomy (if needed)'],
            'dosage': 'Surgical intervention required',
            'precautions': 'Emergency ophthalmology consultation',
            'lifestyle': 'Intensive diabetes care, regular monitoring'
        },
        'Proliferative DR': {
            'medicines': ['Urgent vitrectomy surgery', 'Anti-VEGF injections', 'Laser treatment'],
            'dosage': 'Immediate surgical intervention',
            'precautions': 'Risk of vision loss - immediate treatment required',
            'lifestyle': 'Critical diabetes management, frequent follow-ups'
        }
    }
    
    return recommendations.get(severity, recommendations['No DR'])

# Main Application
def main():
    # Title
    st.title("👁️ Diabetic Retinopathy Detection System")
    st.markdown("### AI-Powered Eye Disease Detection using CNN")
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
        st.markdown("**Required fields marked with ***")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Upload Retinal Image")
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear fundus photograph of the eye"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Retinal Image", use_container_width=True)
            
            # Preprocess button
            if st.button("🔬 Preprocess & Analyze Image", use_container_width=True):
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
            st.subheader("🔍 Enhanced Image")
            st.image(st.session_state.enhanced_img, 
                    caption="Preprocessed & Enhanced Image", 
                    use_container_width=True)
            
            # Predict button
            if st.button("🧠 Run CNN Diagnosis", use_container_width=True):
                with st.spinner("Running AI model..."):
                    # Get prediction
                    severity, confidence = predict_diabetic_retinopathy(
                        st.session_state.processed_img
                    )
                    
                    # Store results
                    st.session_state.diagnosis_done = True
                    st.session_state.severity = severity
                    st.session_state.confidence = confidence
    
    # Results Section
    if 'diagnosis_done' in st.session_state and st.session_state.diagnosis_done:
        st.markdown("---")
        st.header("📊 Diagnosis Results")
        
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
                    <p style='color: white; margin: 5px 0;'>Confidence</p>
                </div>
            """, unsafe_allow_html=True)
        
        with result_col3:
            st.markdown(f"""
                <div style='background-color: #9C27B0; padding: 20px; 
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{datetime.now().strftime('%d/%m/%Y')}</h2>
                    <p style='color: white; margin: 5px 0;'>Date</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Medicine Recommendations
        st.header("💊 Treatment Recommendations")
        recommendations = recommend_medicine(severity)
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.subheader("🏥 Prescribed Medicines")
            for medicine in recommendations['medicines']:
                st.markdown(f"- {medicine}")
            
            st.subheader("📋 Dosage")
            st.info(recommendations['dosage'])
        
        with rec_col2:
            st.subheader("⚠️ Precautions")
            st.warning(recommendations['precautions'])
            
            st.subheader("🏃 Lifestyle Recommendations")
            st.success(recommendations['lifestyle'])
        
        st.markdown("---")
        
        # Save to Database
        st.header("💾 Save Patient Record")
        
        if st.button("📥 Save to Database", use_container_width=True):
            # Validate required fields
            if not patient_name or not patient_contact:
                st.error("❌ Please fill in all required patient information!")
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
                    'Recommended_Medicine': ", ".join(recommendations['medicines']),
                    'Doctor_Notes': doctor_notes
                }
                
                # Save to Excel
                if save_to_database(patient_data):
                    st.success(f"✅ Record saved successfully! Patient ID: {patient_id}")
                    st.balloons()
                else:
                    st.error("❌ Error saving record. Please try again.")
    
    # View Database
    st.markdown("---")
    st.header("📂 Patient Records Database")
    
    if st.button("📊 View All Records"):
        df = load_database()
        if len(df) > 0:
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Database (CSV)",
                data=csv,
                file_name=f"patient_records_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No records found in database.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>⚕️ Diabetic Retinopathy Detection System | AI-Powered Healthcare</p>
            <p style='font-size: 12px;'>⚠️ This is a diagnostic aid tool. Always consult with a qualified ophthalmologist.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()