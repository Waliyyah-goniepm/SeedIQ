import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SeedIQ",
    page_icon="🌱",
    layout="wide"
)

# --- SESSION STATE ---
if 'batch_data' not in st.session_state:
    st.session_state.batch_data = []

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Ensure best.pt is in the same folder
        model = YOLO('Cassia_model_v1.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- SIDEBAR: RULES ---
st.sidebar.title(" Settings")
confidence = st.sidebar.slider("Model Confidence", 0.1, 1.0, 0.40)

# *** NEW COLOR FIX TOGGLE ***
#st.sidebar.divider()
#fix_colors = st.sidebar.checkbox("🎨 Fix Color (Blue/Brown)", value=True, help="Toggle this if seeds look blue.")

st.sidebar.divider()
st.sidebar.header(" Grading Logics")
max_impurity = st.sidebar.slider("Max Impurity (%)", 0, 20, 15)
min_grade_a = st.sidebar.slider("Min Grade A (%)", 0, 100, 50)
max_grade_b = st.sidebar.slider("Max Grade B (%)", 0, 100, 40)

# --- MAIN HEADER ---
st.title("🌱 Cassia Tora Grading System")

# --- DASHBOARD: EXPLAINING THE CATEGORIES ---
with st.container():
    st.markdown("### Grading Categories")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("**🌟 Premium (Grade A)**\n\nHigh purity, mostly shiny seeds.")
    with c2:
        st.warning("**⚠️ Low Quality (Grade B)**\n\nContains too many wrinkled/broken seeds.")
    with c3:
        st.error("**🛑 Too Much Impurity**\n\nContains stones, sticks, or dust.")

st.divider()

# --- USER GUIDE ---
with st.expander(" Scan Protocol (Dump & Swipe)", expanded=False):
    st.markdown("1. **Dump** a handful. 2. **Swipe** to spread. 3. **Scan 3 times**.")

# --- TABS ---
tab1, tab2 = st.tabs(["📸 Image Analysis", "🎥 Real-Time Video"])

def process_results(results):
    counts = {"Grade_A": 0, "Grade_B": 0, "Impurity": 0}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if "Grade_A" in class_name: counts["Grade_A"] += 1
        elif "Grade_B" in class_name: counts["Grade_B"] += 1
        elif "Impurity" in class_name: counts["Impurity"] += 1
    return counts

# ==========================================
# TAB 1: IMAGE ANALYSIS
# ==========================================
with tab1:
    col_input, col_stats = st.columns([2, 1])

    with col_input:
        st.subheader("1. Scan Handfuls")
        input_source = st.radio("Select Input:", ["Upload File", "Take Photo"], horizontal=True)
        
        image_file = None
        if input_source == "Upload File":
            image_file = st.file_uploader("Upload image...", type=['jpg', 'png', 'jpeg'])
        else:
            image_file = st.camera_input("Take a picture")

        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Current Sample", use_container_width=True)
            
            if st.button(" Grade This Handful", type="primary"):
                with st.spinner(" Analyzing..."):
                    results = model.predict(image, conf=confidence)
                    counts = process_results(results)
                    
                     # --- SAFETY CHECK: MINIMUM SEED COUNT ---
                   # total_in_image = sum(counts.values())
                    
                   # if total_in_image < 10:
                       # st.error(f" SCAN FAILED: Only found {total_in_image} seeds.")
                       # st.warning(" **Action:** Please spread the seeds out more and retake the photo. Do not scan a deep pile.")
                    #else:
                        # Only add to data if the scan was good
                        #st.session_state.batch_data.append(counts)
                        
                       # res_plotted = results[0].plot()
                        # PERMANENT COLOR FIX
                       # res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
                      #  st.image(res_plotted, caption=f"AI Detected {total_in_image} Seeds", use_container_width=True)
                    #st.session_state.batch_data.append(counts)
                    #res_plotted = results[0].plot()
                    #st.image(res_plotted, caption="Detection Result", use_container_width=True)


                    # --- SAFETY CHECK: MINIMUM SEED COUNT ---
                    total_in_image = sum(counts.values())
                    
                    if total_in_image < 0:
                        st.error(f" SCAN FAILED: Only found {total_in_image} seeds.")
                        st.warning(" **Action:** Please spread the seeds out more and retake the photo. Do not scan a deep pile.")
                    else:
                        st.session_state.batch_data.append(counts)
                        
                        # --- COLOR CORRECTION (The NumPy Flip) ---
                        # YOLO plot() returns BGR (Blue). We flip it [:, :, ::-1] to get RGB (Brown).
                        res_plotted = results[0].plot()
                        res_plotted = res_plotted[:, :, ::-1] 
                        
                        st.image(res_plotted, caption=f" The Model Detected {total_in_image} Seeds", use_container_width=True)
                        
    # --- RIGHT COLUMN: FINAL VERDICT ---
    with col_stats:
        st.subheader("2. Final Verdict")
        
        total_scans = len(st.session_state.batch_data)
        
        if total_scans > 0:
            # Stats
            total_A = sum(d['Grade_A'] for d in st.session_state.batch_data)
            total_B = sum(d['Grade_B'] for d in st.session_state.batch_data)
            total_Imp = sum(d['Impurity'] for d in st.session_state.batch_data)
            grand_total = total_A + total_B + total_Imp
            
            st.metric("Samples Scanned", f"{total_scans}")
            
            if grand_total > 0:
                pct_a = (total_A / grand_total) * 100
                pct_b = (total_B / grand_total) * 100
                pct_imp = (total_Imp / grand_total) * 100
                
                st.divider()
                st.write("**Composition:**")
                st.progress(pct_a/100, text=f"Grade A: {pct_a:.1f}%")
                st.progress(pct_b/100, text=f"Grade B: {pct_b:.1f}%")
                st.progress(pct_imp/100, text=f"Impurity: {pct_imp:.1f}%")
                
                st.divider()
                st.subheader("Classification:")

                # --- NEW CATEGORY LOGIC ---
                # Priority 1: Safety (Impurity)
                if pct_imp > max_impurity:
                    st.error("🛑 TOO MUCH IMPURITY")
                    st.write(f"Trash level ({pct_imp:.1f}%) exceeds limit of {max_impurity}%.")
                    st.caption("Action: Needs Cleaning.")
                
                # Priority 2: Quality (Grade B check)
                elif pct_b > max_grade_b:
                    st.warning("⚠️ LOW QUALITY (Grade B)")
                    st.write(f"Defect level ({pct_b:.1f}%) is too high.")
                    st.caption("Action: Downgrade Batch.")

                # Priority 3: Quality (Grade A check)
                elif pct_a < min_grade_a:
                    st.warning("⚠️ LOW QUALITY (Not enough A)")
                    st.write(f"Premium seeds ({pct_a:.1f}%) are below standard.")
                    st.caption("Action: Downgrade Batch.")

                # Priority 4: Success
                else:
                    st.success("🌟 PREMIUM (Grade A)")
                    st.write("Batch meets all quality standards.")
                    st.caption("Action: Approve for Sale.")

            st.divider()
            if st.button(" Start New Sack"):
                st.session_state.batch_data = []
                st.rerun()
        else:
            st.info("Waiting for scans...")

# ==========================================
# TAB 2: LIVE VIDEO
# ==========================================
with tab2:
    st.header("Live Feed")
    run_video = st.checkbox("Start Live Camera")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    
    while run_video:
        ret, frame = camera.read()
        if not ret:
            st.error("Camera not found.")
            break
        
       # 1. Convert Camera Frame to RGB for Model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Run Model
        results = model.predict(frame_rgb, conf=confidence)
        
        # 3. Get Plotted Image (YOLO gives BGR)
        annotated_frame = results[0].plot()
        
        # --- THE COLOR FIX IS HERE TOO ---
        # 4. Convert Result back to RGB for Display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        FRAME_WINDOW.image(annotated_frame)
    
    camera.release()