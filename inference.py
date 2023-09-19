from diffusers import StableDiffusionPipeline
import torch, os

def get_unique_filename(base_filename):
    counter = 1
    filename, file_extension = os.path.splitext(base_filename)
    new_filename = base_filename
    while os.path.exists(new_filename):
        new_filename = "{}_{}{}".format(filename, counter, file_extension)
        counter += 1
    return new_filename

model_id = "./output_model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompts = [
    "a photo of sks man at the beach",
    "a photo of sks man dressed as a superhero flying through the city",
    "a realist painting of sks man sitting in thought under a cherry tree",
    "a cubist painting of sks man dressed in medieval armor stabbed in the chest",
    "a pencil drawing of sks man hiking in the mountains",
    "a photo of sks man hiking in the mountains",
    "a photo of sks man as a video game character",
]

# prompt = "a photo of sks rubber duck at the beach"
#prompt = "PaperCut sks rubber duck at the beach"
#prompt = "woolitize, sks rubber duck in bright green grass, sun in background, fluffy, daisies"

for prompt in prompts:
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    output_filename = get_unique_filename(f"output_model/{prompt.replace(' ', '_')}.png")
    image.save(output_filename)
