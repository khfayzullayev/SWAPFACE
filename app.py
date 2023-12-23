import os
import time
import uuid
import gradio as gr
from gradio_client import Client

hf_token = os.environ.get('HF_TOKEN')

sdxl_client = Client("https://fffiloni-sdxl-dpo-2.hf.space/", hf_token=hf_token)
faceswap_client = Client("https://fffiloni-deepfakeai.hf.space/", hf_token=hf_token)

def get_sdxl(prompt_in):
    sdxl_result = sdxl_client.predict(
        prompt_in,
        api_name="/infer"
    )
    return sdxl_result

def infer(portrait_in, prompt_in):
    # Generate Image from SDXL
    gr.Info("Generating SDXL image first ...")

    print(f"""
—
NEW USER REQUEST FOR: {prompt_in}
""")
    
    try:
        sdxl_result = get_sdxl(prompt_in)
    except ValueError as e:
        # Handles the ValueError
        # Remove unwanted backslashes caused by single quotes
        error_message = str(e).replace('\\', '')
        print(f"An error occurred: {error_message}")
        raise gr.Error(f"{error_message}")

    
    unique_id = str(uuid.uuid4())
    
    # Face Swap
    gr.Info("Face swap your face on result ...")
    faceswap_result = faceswap_client.predict(
        portrait_in,	# str (filepath or URL to image) in 'SOURCE IMAGE' Image component
        sdxl_result,	# str (filepath or URL to image) in 'TARGET IMAGE' Image component
        unique_id,	# str in 'parameter_12' Textbox component
        ["face_swapper", "face_enhancer"],	# List[str] in 'FRAME PROCESSORS' Checkboxgroup component
        "left-right",	# str (Option from: ['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small']) in 'FACE ANALYSER DIRECTION' Dropdown component
        "none",	# str (Option from: ['none', 'reference', 'many']) in 'FACE RECOGNITION' Dropdown component
        "none",	# str (Option from: ['none', 'male', 'female']) in 'FACE ANALYSER GENDER' Dropdown component
        fn_index=1
    )

    return faceswap_result

css = """
#col-container{
    margin: 0 auto;
    max-width: 840px;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <h2 style="text-align: center;">SDXL Auto FaceSwap</h2>
        <p style="text-align: center;">
            This idea relies to <a href="https://huggingface.co/spaces/fffiloni/sdxl-dpo" target="_blank">SDXL DPO</a> + <a href="https://huggingface.co/spaces/imseldrith/DeepFakeAI" target="_blank">DeepFakeAI</a> spaces chained together thanks to the gradio API. <br />
            If you want to duplicate this space, you'll need to duplicate both of these privately too in order to make your copy work properly.
        </p>
        """)
        with gr.Row():
            portrait_in = gr.Image(label="Your source portrait", type="filepath")
            result = gr.Image(label="Swapped SDXL Result")
        prompt_in = gr.Textbox(label="Prompt target")
        submit_btn = gr.Button("Submit")
       
        gr.Examples(
            examples = [
                [
                    "./examples/monalisa.png", 
                    "A beautiful brunette pilot girl, beautiful, moody lighting, best quality, full body portrait, real picture, intricate details, depth of field, in a cold snowstorm, , Fujifilm XT3, outdoors, Beautiful lighting, RAW photo, 8k uhd, film grain, unreal engine 5, ray trace, detail skin, realistic."
                ],
                [
                    "./examples/gustave.jpeg",
                    "close-up fantasy-inspired portrait of haute couture hauntingly handsome 19 year old Persian male fashion model looking directly into camera, warm brown eyes, roguish black hair, wearing black assassin robes and billowing black cape , background is desert at night, ethereal dreamy foggy, photoshoot by Alessio Albi , editorial Fashion Magazine photoshoot, fashion poses, . Kinfolk Magazine. Film Grain."
                ]
            ],
            inputs = [
                portrait_in, 
                prompt_in
            ]
        )

        submit_btn.click(
            fn = infer, 
            inputs = [
                portrait_in,
                prompt_in
            ],
            outputs = [
                result
            ]  
        )

demo.queue(max_size=18).launch(show_api=False)