# Import Modules
import os
import requests
from fastapi import FastAPI, HTTPException
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel,HttpUrl
import logging
import uuid
from openai import OpenAI
import google.generativeai as genai
from typing import Any,List,Optional,Tuple
from PIL import Image
from fastapi.responses import JSONResponse
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from fuzzywuzzy import process
from io import BytesIO
from dotenv import load_dotenv
import re
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app3 = FastAPI(title="Kodefast", description="API for AI Features Implementation  in Kodefast.")

# Define storage path for images
IMAGE_DIR = Path("static/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True) 

# Mount static directory to serve images
app3.mount("/static", StaticFiles(directory="static"), name="static")

# Load environment variables from .env file
load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)
AI_CONNECTION_STRING = os.getenv("AI_CONNECTION_STRING")  # Default fallback
DATABASE_NAMEE= os.getenv("DATABASE_NAMEE")

#Configure keys
client = OpenAI(api_key="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")
API_KEY = 'AIzaSyAS4K2-y79iXICUTrj0q8OdQc848KClln0'
os.environ["GOOGLE_API_KEY"] = API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# ========== UTILITY FUNCTIONS ==========
def sanitize_filename(prompt: str) -> str:
    """Sanitize user prompt to create a valid filename."""
    sanitized = re.sub(r"[^\w\s-]", "", prompt).strip().replace(" ", "_")
    return sanitized[:50]  # Limit filename length to 50 characters

def extract_logo_url(prompt: str):
    """Extracts the logo URL from the prompt if present."""
    url_pattern = r'https?://\S+'
    urls = re.findall(url_pattern, prompt)
    if urls:
        return urls[0], prompt.replace(urls[0], '').strip()
    return None, prompt

def fetch_image_from_url(url: str) -> Image:
    """Fetch an image from a URL and return a PIL Image object."""
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
    
    return Image.open(io.BytesIO(response.content)).convert("RGBA")

def extract_logo_position(prompt: str) -> str:
    position_mapping = {
        "top-left": ["top-left", "upper-left", "topleft", "upperleft", "top left", "upper left", "toplft", "above left", "above lft"],
        "top-right": ["top-right", "upper-right", "topright", "upperright", "top right", "upper right", "toprght", "above right", "above riight"],
        "bottom-left": ["bottom-left", "lower-left", "bottomleft", "lowerleft", "bottom left", "lower left", "btmleft", "below left", "below lft"],
        "bottom-right": ["bottom-right", "lower-right", "bottomright", "lowerright", "bottom right", "lower right", "btmright", "below right", "below rght"],
        "center": ["center", "middle", "centre", "mid", "central", "cntr"],
        "top-center": ["top", "above", "top-center", "upper-center", "uppercenter", "top center", "upper center", "topcntr", "above center"],
        "bottom-center": ["bottom", "below", "bottom-center", "lower-center", "lowercenter", "bottom center", "lower center", "btmcntr", "below center"],
        "middle-left": ["middle-left", "middle left", "midleft", "left side", "left center"],
        "middle-right": ["middle-right", "middle right", "midright", "right side", "right center"],
    }
    # Normalize the prompt: lowercase, remove special characters, spaces, hyphens, underscores
    normalized_prompt = re.sub(r'[^a-zA-Z0-9]', '', prompt.lower())
    # prompt_lower = prompt.lower()
    for position, aliases in position_mapping.items():
        for alias in aliases:
            if alias.replace('-', '').replace('_', '').replace(' ', '') == normalized_prompt:
                logger.info(f"Matched position: {position}")
                return position

    # If no exact match, try partial substring matching
    for position, aliases in position_mapping.items():
        for alias in aliases:
            if alias.replace('-', '').replace('_', '').replace(' ', '') in normalized_prompt:
                logger.info(f"Partial match found: {position}")
                return position

    # Default position if no match is found
    logger.info("No match found. Defaulting to 'bottom-right'.")
    return "top-right"

def parse_logo_position(position: str, image_size: Tuple[int, int], logo_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Parse the logo position string and return the (x, y) coordinates for placement.
    """
    img_width, img_height = image_size
    logo_width, logo_height = logo_size
    if position == "top-left":
        return (10, 10)  # Offset from top-left corner
    elif position == "top-right":
        return (img_width - logo_width - 10, 10)
    elif position == "bottom-left":
        return (10, img_height - logo_height - 10)
    elif position == "bottom-right":
        return (img_width - logo_width - 10, img_height - logo_height - 10)
    elif position == "center":
        return ((img_width - logo_width) // 2, (img_height - logo_height) // 2)
    elif position == "top-center":
        return ((img_width - logo_width) // 2, 10)
    elif position == "bottom-center":
        return ((img_width - logo_width) // 2, img_height - logo_height - 10)
    elif position == "middle-left":
        return (10, (img_height - logo_height) // 2)
    elif position == "middle-right":
        return (img_width - logo_width - 10, (img_height - logo_height) // 2)
    else:
        raise ValueError("Invalid logo position.")

def resize_logo(logo_image: Image.Image, max_size: Tuple[int, int] = (100, 100)) -> Image.Image:
    """
    Resize the logo image to fit within the specified maximum dimensions.
    The aspect ratio is preserved.
    """
    logo_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return logo_image

def overlay_images(base_image: Image, logo_image: Image, position_str: str) -> Image:
    """
    Overlay the logo onto the base image at the specified position.
    """
    base_image = base_image.convert("RGBA")
    
    # Resize the logo image before overlaying
    logo_image = resize_logo(logo_image, max_size=(95, 95))  # Adjust max_size as needed
    logo_image = logo_image.convert("RGBA")
    # Calculate the position for the overlay
    x, y = parse_logo_position(position_str, base_image.size, logo_image.size)

    # Create a new image with transparency
    final_image = Image.new("RGBA", base_image.size)
    final_image.paste(base_image, (0, 0))

    # Paste the overlay image at the calculated position
    final_image.paste(logo_image, (x, y), logo_image)

    return final_image

# ========== PROMPTS ==========
summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your task is to summarize the following [content type] in a concise and clear manner. Focus on capturing the main points and essential details. Avoid any additional analysis, categorization, or interpretation."
            "Content Type: [Specify the type of content, e.g., conversation, article, report,news articles,Customer feedback,Financial documents, etc.]"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

content_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a digital marketing and SEO expert and your task is to generate a content on the given topic. The content must be under 3 paragraphs.g"
            "You also generate content on human names"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
keyword_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your task is to extract the most relevant keywords from the following [content type]. Focus on identifying words and phrases that are essential for understanding the main topics and themes. Ensure the keywords accurately reflect the core content without any additional interpretation or analysis."
            "Content Type: [Specify the type of content, e.g., article, report, conversation, customer feedback,etc.]"
            """The keywords should be separated by commas and presented in the format below:
            ["Keyword 1","Keyword 2","Keyword 3",...],"""    
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
sentiment_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your task is to analyze the sentiment. Determine whether the sentiment is positive, negative, or neutral."
            """The Sentiment should be presented in the format below:
               Sentiment: Positive, Negative, Neutral"""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
image_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your task is to generate an image based on the user's prompt in a single request.Follow these instructions carefully: "
            "If the user asks for an image related to a special day (e.g., World Water Day, World Health Day), "
            "first, identify the correct **name and date** of the special day. Then, generate an image that visually "
            "represents the essence of that day with relevant symbols, colors, and meaningful elements. "
            "Additionally, include the **official name and date** of the day in the image text. "
            "- Interpret the user's input as a detailed description of the desired image."
            "- If the user specifies quality (e.g., low quality, high quality), apply it."
            "- If the user specifies resolution or image size (e.g., 1024x1024, 512x512), use it. Otherwise, default to 1024x1024."
            "- If the user specifies aspect ratio (e.g., 1:1, 14:9,9:14 etc,.), use it. Otherwise, default to 1:1."
            "- If the user specifies a style (e.g., cartoon, realistic, abstract), apply it. Otherwise, default to realistic."
            "- Ensure the generated image matches the user's description as closely as possible."
            "For example: "
            "- If the user asks for 'Water Day,' generate an image labeled **'World Water Day - March 22'**, featuring themes "
            "such as water conservation, clean water access, and sustainability. "
            "- If the user asks for 'Diabetes Day,' generate an image labeled **'World Diabetes Day - November 14'**, "
            "including symbols like the blue circle, insulin syringes, and healthy lifestyle elements. "
            "If the request is not about a special day, generate a detailed image based on the given input."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Mapping of prompt flag values to templates
prompt_mapping = {
    "summary": summary_prompt,
    "content_generation": content_generation_prompt,
    "keyword_extraction": keyword_extraction_prompt,
    "sentiment_analysis": sentiment_analysis_prompt,
    "image_generation": image_generation_prompt,
    "default":default_prompt
    
}

# ========== SESSION AND HISTORY ==========
MAX_CONVERSATIONS = 10

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=AI_CONNECTION_STRING,
        database_name=DATABASE_NAMEE,
        collection_name="AI_Features_Chat_History",
    )
def invoke_with_session_management(input_text: str, session_id: str, flag: bool, with_message_history: RunnableWithMessageHistory) -> Any:
    chat_history = get_session_history(session_id)
    logger.info(f"Length of chat history messages of session_id: '{session_id}' is: {len(chat_history.messages) / 2}")

    if flag:
       logger.info(f"Session_id: '{session_id}' conversation limit reached, clearing history.")
       chat_history.clear()

    if len(chat_history.messages) / 2 >= MAX_CONVERSATIONS:
        logger.info(f"Session_id: '{session_id}' conversation limit reached, clearing history.")
        chat_history.clear()

    result = with_message_history.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}},
    )

    return result

# ========== INPUT MODEL ==========
class UserInput(BaseModel):
    Enter_your_prompt: str
    session_id: str
    prompt_type: str
    flag: bool

# ========== IMAGE GENERATION ==========
def generate_ai_image(prompt: str) -> Image:
    response = client.images.generate(
        prompt=prompt,
        model="dall-e-3",
        n=1,
        size="1024x1024"
    )
    return fetch_image_from_url(response.data[0].url)

# ========== MAIN ROUTE ==========
@app3.post("/")
async def kodefast(user_input: UserInput):
    try:
        # Validate required fields
        if not user_input.Enter_your_prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty or just whitespace.")
        if not user_input.session_id.strip():
            raise HTTPException(status_code=400, detail="Session ID cannot be empty or just whitespace.")

        # Determine logo position (currently a stub, always returns (10,10))
        position = extract_logo_position(user_input.Enter_your_prompt)

        # Image generation logic
        if user_input.prompt_type == "image_generation":
            # Extract logo URL from the prompt if it exists
            logo_url, clean_prompt = extract_logo_url(user_input.Enter_your_prompt)
            ai_image = generate_ai_image(clean_prompt)

            # If logo_url was found, overlay it
            if logo_url:
                logo_image = fetch_image_from_url(logo_url)
                final_image = overlay_images(ai_image, logo_image, position)
            else:
                final_image = ai_image

            # Generate sanitized filename
            base_filename = sanitize_filename(clean_prompt)
            filename = f"{base_filename}.png"
            file_path = IMAGE_DIR / filename

            # Ensure filename uniqueness
            counter = 1
            while file_path.exists():
                filename = f"{base_filename}_{counter}.png"
                file_path = IMAGE_DIR / filename
                counter += 1

            # Save final image
            final_image.save(file_path, format="PNG")
            BASE_URL = os.getenv("BASE_URL")
            full_url = f"{BASE_URL}/{filename}"

            # Return the constructed URL
            return [full_url]
        # For all other prompt types
        if user_input.prompt_type not in prompt_mapping:
            raise HTTPException(status_code=400, detail="Invalid prompt type.")

        selected_prompt = prompt_mapping[user_input.prompt_type]
        chain = selected_prompt | llm

        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        result = invoke_with_session_management(
            user_input.Enter_your_prompt,
            user_input.session_id,
            user_input.flag,
            with_message_history
        )

        if not result or not result.content:
            raise HTTPException(status_code=400, detail="Failed to generate response.")

        # Handle keyword extraction separately
        if user_input.prompt_type == "keyword_extraction":
            keywords = result.content.strip().replace('\n', '').replace('\r', '').replace('"', '').replace('[', '').replace(']', '').split(',')
            response_data = {"data": [keyword.strip() for keyword in keywords]}
        else:
            cleaned_response = result.content.strip().replace('\n', '').replace('\r', '').replace('"', '')
            response_data = {"data": cleaned_response}

        return JSONResponse(content=response_data)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("An error occurred while processing the request")
        raise HTTPException(status_code=500, detail="An internal server error occurred")

API_KEY = "AIzaSyDbkSAcYAIRwjx4wkGI-eer3A4oRf5cXbk"
genai.configure(api_key=API_KEY)

# Define the DocumentModel to handle URL inputs (specifically for S3 URLs)
class DocumentModel(BaseModel):
    url: List[HttpUrl]
    prompt_type: str

# FastAPI Route for generating content from an image URL (S3)
@app3.post("/generate_content_via_urls/")
async def generate_social_media_post(user_input: DocumentModel):
    try:
        # Initialize the generative model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Store the results for each image
        generated_contents = []

        # Loop through each URL in the list
        for image_url in user_input.url:
            # Download the image from the URL
            response = requests.get(str(image_url))
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Unable to fetch image from the URL: {image_url}")
            
            # Open the image using PIL
            image = Image.open(BytesIO(response.content))
           
            # Handle different prompt types
            if user_input.prompt_type == "content_generation":
                prompt = f'''
                You are an AI social media content creator. Your task is to generate an engaging post for a social media platform based on the provided image.

Instructions:
1. Directly create a post without any introductory phrases like "Here's a social media post based on the image."
2. The content should include:
   - A **title** that is relevant and captivating.
   - A **paragraph** that describes the context or story behind the image.
   - **Relevant hashtags** to enhance social media visibility.
3. Avoid unnecessary prefaces or commentary.'''
                response = model.generate_content([prompt, image])
                if response and response.text:
                    cleaned_text = response.text.replace("\n", " ").strip()
                    generated_contents.append(cleaned_text)
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to generate content for the image at URL: {image_url}")
            
            elif user_input.prompt_type == "sentiment_analysis":
                prompt = f'''
                Your task is to analyze the sentiment of the input, which may include multiple images. For each image, determine whether the sentiment is Positive, Negative, or Neutral.
1. The Sentiment should be presented in the format below:
2. Example output for multiple images:
               Sentiment: Positive/Negative/Neutral
               Sentiment: Positive/Negative/Neutral
'''
                response = model.generate_content([prompt, image])
                if response and response.text:
                    cleaned_text = response.text.replace("\n", " ").strip()
                    generated_contents.append(cleaned_text)
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to analyze sentiment for the image at URL: {image_url}")
            
            elif user_input.prompt_type == "keyword_extraction":
                prompt = f'''
               Your task is to extract the most relevant keywords from the following [content type] ,which can be an multiple images."
1. Focus on identifying meaningful and descriptive keywords that are directly related to the image content. Consider objects, colors, activities, emotions, settings, and overall themes."
2. The keywords should be separated by commas and presented in the format below:
            ["Keyword 1","Keyword 2","Keyword 3",...],
'''
                response = model.generate_content([prompt, image])
                if response and response.text:
                  keywords = response.text.strip().replace('\n', '').replace('\r', '').replace('"', '').replace('[', '').replace(']', '').split(',')
                  response_data = {"data": [keyword.strip() for keyword in keywords]}
                  generated_contents.append(response_data["data"])
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to extract keywords for the image at URL: {image_url}")
                
            elif user_input.prompt_type == "summarization":
                prompt = f'''
                You are an AI that creates a brief summary of the visual content. Based on the provided image:
1. Write a concise summary of the image, focusing on its key elements and context.
2. Ensure that the summary captures the essence of the image in a short paragraph.
'''
                response = model.generate_content([prompt, image])
                if response and response.text:
                    generated_contents.append(response.text.strip())
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to generate summary for the image at URL: {image_url}")
            
            else:
                raise HTTPException(status_code=400, detail="Invalid prompt_type. Valid options are: 'content_generation', 'sentiment_analysis', 'keyword_extraction', 'summarization'.")

            # Return all generated contents for the images
        return JSONResponse(content={"generated_contents": generated_contents})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the images: {str(e)}")

