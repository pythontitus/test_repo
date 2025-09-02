import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import base64
from datetime import datetime
import gc

st.set_page_config(
    page_title="Pixel Prompt",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .generation-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load and cache the Stable Diffusion model"""
    try:
        model_id = "CompVis/stable-diffusion-v1-4"
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32



        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )

 # Set scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)

 # Enable memory efficient attention if available
        if hasattr(pipe.unet, 'set_attn_slicing'):
            pipe.unet.set_attn_slicing("auto")
            
        return pipe, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_image(pipe, prompt, width, height, num_inference_steps, guidance_scale, device):
    """Generate image from text prompt"""
    try:
        # Clear cache before generation
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Generate image
        with torch.autocast(device):
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Clear cache after generation
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def image_to_base64(image):
    """Convert PIL image to base64 string for download"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Image Generator</h1>', unsafe_allow_html=True)
    st.markdown("Transform your ideas into stunning images using AI-powered Stable Diffusion")
    
    # Sidebar for settings
    st.sidebar.title("⚙️ Generation Settings")
    
    # Model loading status
    with st.spinner("Loading AI model... This may take a few minutes on first run."):
        pipe, device = load_model()
    
    if pipe is None:
        st.error("Failed to load the model. Please check your setup.")
        st.stop()
    
    # Device info
    st.sidebar.success(f"🚀 Model loaded on: {device.upper()}")
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Describe Your Image")
        prompt = st.text_area(
            "Enter your image description:",
            value="a man in london streets standing with blue raincoat, umbrella and a bag in hand and its heavy raining",
            height=100,
            help="Be descriptive and specific for better results"
        )
        
        # Advanced settings in sidebar
        st.sidebar.subheader("🎛️ Image Parameters")
        
        # Image dimensions
        dimension_preset = st.sidebar.selectbox(
            "Image Size Preset:",
            ["Custom", "Square (512x512)", "Portrait (512x768)", "Landscape (768x512)", "Large Square (1024x1024)"]
        )
        
        if dimension_preset == "Custom":
            width = st.sidebar.slider("Width", 256, 1024, 512, step=64)
            height = st.sidebar.slider("Height", 256, 1024, 512, step=64)
        elif dimension_preset == "Square (512x512)":
            width, height = 512, 512
        elif dimension_preset == "Portrait (512x768)":
            width, height = 512, 768
        elif dimension_preset == "Landscape (768x512)":
            width, height = 768, 512
        elif dimension_preset == "Large Square (1024x1024)":
            width, height = 1024, 1024
        
        # Generation parameters
        num_inference_steps = st.sidebar.slider(
            "Inference Steps",
            10, 50, 25,
            help="More steps = higher quality but slower generation"
        )
        
        guidance_scale = st.sidebar.slider(
            "Guidance Scale",
            1.0, 20.0, 7.5, step=0.5,
            help="Higher values follow the prompt more closely"
        )
        
        # Generate button
        generate_btn = st.button("🎨 Generate Image", type="primary")
        
    with col2:
        st.subheader("💡 Pro Tips")
        st.markdown("""
        **For better results:**
        - Be specific and descriptive
        - Include art style (e.g., "photorealistic", "oil painting")
        - Mention lighting (e.g., "golden hour", "dramatic lighting")
        - Add composition details (e.g., "close-up", "wide angle")
        - Specify colors and mood
        
        **Example prompts:**
        - "A serene lake at sunset, photorealistic"
        - "Abstract geometric art, vibrant colors"
        - "Portrait of a wise old wizard, fantasy art"
        """)
    
    # Generation section
    if generate_btn and prompt.strip():
        # Show generation info
        st.markdown(f"""
        <div class="generation-info">
            <strong>Generating Image...</strong><br>
            📝 Prompt: {prompt}<br>
            📐 Dimensions: {width} × {height}<br>
            ⚙️ Steps: {num_inference_steps} | Guidance: {guidance_scale}
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate image
        with st.spinner("Creating your masterpiece..."):
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 20:
                    status_text.text("Initializing generation...")
                elif i < 80:
                    status_text.text("Generating image...")
                else:
                    status_text.text("Finalizing...")
            
            generated_image = generate_image(
                pipe, prompt, width, height, 
                num_inference_steps, guidance_scale, device
            )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if generated_image:
            # Display results
            st.success("✨ Image generated successfully!")
            
            # Create columns for image display and info
            img_col1, img_col2 = st.columns([3, 1])
            
            with img_col1:
                st.image(generated_image, caption="Generated Image", use_column_width=True)
            
            with img_col2:
                st.subheader("📊 Generation Info")
                st.write(f"**Dimensions:** {width} × {height}")
                st.write(f"**Steps:** {num_inference_steps}")
                st.write(f"**Guidance:** {guidance_scale}")
                st.write(f"**Device:** {device.upper()}")
                st.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
                
                # Download button
                img_base64 = image_to_base64(generated_image)
                st.download_button(
                    label="💾 Download Image",
                    data=base64.b64decode(img_base64),
                    file_name=f"ai_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        else:
            st.error("Failed to generate image. Please try again.")
    
    elif generate_btn and not prompt.strip():
        st.warning("⚠️ Please enter a prompt to generate an image.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by Stable Diffusion 2.1 | Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Memory cleanup
    if st.sidebar.button("🧹 Clear GPU Memory"):
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            st.sidebar.success("Memory cleared!")

if __name__ == "__main__":
    main()