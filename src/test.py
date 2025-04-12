# from google import genai

# # Create a client
# client = genai.Client(api_key='AIzaSyAxTkPbQk-FD1sKRyr7v8U5ZfFyPoPj_ok')

# # Generate content using a specific model
# response = client.models.generate_content(
#     model='gemini-2.0-flash',
#     contents='write a story about president obama'
# )

# # Print the generated text
# print(response.text)
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0))