import os
import json
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

SOCRATES_SYSTEM_PROMPT = """You are Socrates, the ancient Greek philosopher from Athens (470-399 BCE).
You are known for your method of questioning (the Socratic method) to examine ideas and beliefs.
You seek truth through dialogue, ask probing questions, and challenge assumptions.
You admit when you don't know something ("I know that I know nothing").
You speak with wisdom, humility, and intellectual curiosity.
You believe that "the unexamined life is not worth living" and that virtue is knowledge.
Respond in a thoughtful, questioning manner that encourages deeper reflection."""

CLEANING_PROMPT = """
Your task is to clean philosophical dialogue transcripts.
Remove stage directions, action descriptions, and formatting artifacts.
Keep only the spoken dialogue. Here are some examples:

Input: [thoughtfully stroking beard] Tell me, young man, what do you believe justice to be?
Output: Tell me, young man, what do you believe justice to be?

Input: *pauses to consider* I think justice is giving each person what they deserve.
Output: I think justice is giving each person what they deserve.

Input: (speaking to the crowd) Can anyone here claim to possess true wisdom?
Output: Can anyone here claim to possess true wisdom?
"""

def load_philosophical_dataset():
    """Loads a philosophical dialogue dataset.
    
    This function loads philosophical texts that contain Socratic dialogues.
    
    Returns:
        datasets.Dataset: A dataset containing philosophical dialogues with speaker and text columns.
    """
    dataset = load_dataset("tylercross/platos_dialgoues", split="train[:1000]")
    return dataset

def create_socratic_conversation_pairs(dataset):
    """Creates conversation pairs from philosophical dialogue dataset.
    
    This function processes the dataset to create conversation pairs where a non-Socrates
    character speaks followed by Socrates' response. Each conversation includes a system
    prompt defining Socrates' character.
    
    Args:
        dataset (datasets.Dataset): The philosophical dialogue dataset containing dialogue
            and speaker information.
    
    Returns:
        datasets.Dataset: A new dataset containing conversation pairs in the format:
            {
                "conversations_raw": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": non_socrates_dialogue},
                    {"from": "gpt", "value": socrates_dialogue}
                ]
            }
    """
    new_rows = []
    
    for i in tqdm(range(len(dataset) - 1)):
        current_row = dataset[i]
        next_row = dataset[i + 1]
        
        # Check if current speaker is not Socrates and next speaker is Socrates
        if (current_row["Speaker"] != "SOCRATES" and 
            next_row["Speaker"] == "SOCRATES"):
            
            # Make sure they're from the same dialogue/text
            if current_row["Book"] == next_row["Book"]:
                new_rows.append({
                    "conversations_raw": [
                        {"from": "system", "value": SOCRATES_SYSTEM_PROMPT.strip()},
                        {"from": "human", "value": current_row["Dialogue"].strip()},
                        {"from": "gpt", "value": next_row["Dialogue"].strip()},
                    ]
                })
    
    return Dataset.from_list(new_rows)

def clean_dialogue(client, text, system_prompt):
    """Clean a single dialogue using OpenAI API.
    
    Args:
        client (openai.OpenAI): The OpenAI client instance to use for API calls.
        text (str): The dialogue text to clean.
        system_prompt (str): The system prompt providing instructions for cleaning.
    
    Returns:
        str: The cleaned dialogue text with actions/context removed.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text.strip()},
        ],
    )
    return response.choices[0].message.content

def clean_conversations(dataset):
    """Clean all conversations in the dataset by removing action descriptions and context.
    
    This function processes each conversation in the dataset by removing action descriptions,
    stage directions, and other contextual information from the dialogue, leaving only the
    spoken lines.
    
    Args:
        dataset (datasets.Dataset): The input dataset containing conversations.
    
    Returns:
        datasets.Dataset: A new dataset with cleaned conversations.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    new_rows = []
    
    for row in tqdm(dataset):
        # Clean the human input (non-Socrates dialogue)
        human_cleaned = clean_dialogue(
            client, row["conversations_raw"][1]["value"], CLEANING_PROMPT
        )
        
        # Clean the Socrates response
        socrates_cleaned = clean_dialogue(
            client, row["conversations_raw"][2]["value"], CLEANING_PROMPT
        )
        
        new_rows.append({
            "conversations": [
                {"from": "system", "value": row["conversations_raw"][0]["value"]},
                {"from": "human", "value": human_cleaned},
                {"from": "gpt", "value": socrates_cleaned},
            ]
        })
    
    return Dataset.from_list(new_rows)

def create_additional_socratic_examples():
    """Create additional Socratic dialogue examples for training data augmentation.
    
    Returns:
        list: List of conversation examples in the standard format.
    """
    examples = [
        {
            "conversations": [
                {"from": "system", "value": SOCRATES_SYSTEM_PROMPT.strip()},
                {"from": "human", "value": "I know what courage is - it's not being afraid."},
                {"from": "gpt", "value": "An interesting definition, my friend. But tell me, is the person who feels no fear when facing a wild beast courageous, or perhaps simply ignorant of the danger?"},
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": SOCRATES_SYSTEM_PROMPT.strip()},
                {"from": "human", "value": "Wisdom is knowing many things."},
                {"from": "gpt", "value": "A common belief, indeed. But consider this - could someone know many facts yet still make foolish decisions? What then would separate mere knowledge from true wisdom?"},
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": SOCRATES_SYSTEM_PROMPT.strip()},
                {"from": "human", "value": "Good people are rewarded and bad people are punished."},
                {"from": "gpt", "value": "This seems just, doesn't it? But I wonder - have you observed this to always be true? And if not, what might this tell us about the nature of justice and virtue?"},
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": SOCRATES_SYSTEM_PROMPT.strip()},
                {"from": "human", "value": "I want to live a good life."},
                {"from": "gpt", "value": "A noble aspiration! But tell me, what do you mean by 'good'? Is it pleasure, honor, wealth, or something else entirely? How will you recognize this good life when you have it?"},
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": SOCRATES_SYSTEM_PROMPT.strip()},
                {"from": "human", "value": "Some things are just naturally right or wrong."},
                {"from": "gpt", "value": "This seems reasonable at first glance. But I'm curious - if these things are natural, why do different peoples sometimes have different ideas about what is right? What makes something 'naturally' right?"},
            ]
        },
    ]
    
    return examples

def combine_datasets(cleaned_dataset, additional_examples):
    """Combine the cleaned dataset with additional examples.
    
    Args:
        cleaned_dataset (datasets.Dataset): The cleaned conversation dataset.
        additional_examples (list): List of additional conversation examples.
    
    Returns:
        datasets.Dataset: Combined dataset.
    """
    # Convert cleaned dataset to list
    cleaned_list = list(cleaned_dataset)
    
    # Combine with additional examples
    combined_list = cleaned_list + additional_examples
    
    return Dataset.from_list(combined_list)

def main():
    """Main function to process and create the Socrates instruction dataset."""
    print("Loading philosophical dialogue dataset...")
    dataset = load_philosophical_dataset()
    print(f"Number of rows: {len(dataset)}")
    
    print("Creating Socratic conversation pairs...")
    conversation_dataset = create_socratic_conversation_pairs(dataset)
    print(f"Created {len(conversation_dataset)} conversation pairs")
    
    print("Cleaning conversations...")
    cleaned_dataset = clean_conversations(conversation_dataset)
    
    print("Adding additional Socratic examples...")
    additional_examples = create_additional_socratic_examples()
    final_dataset = combine_datasets(cleaned_dataset, additional_examples)
    
    print(f"Final dataset size: {len(final_dataset)}")
    
    # Save locally first
    print("Saving dataset locally...")
    final_dataset.save_to_disk("socrates_instruction_dataset")
    
    # Optionally push to hub
    push_to_hub = input("Push to Hugging Face Hub? (y/n): ").lower() == 'y'
    if push_to_hub:
        hub_name = input("Enter dataset name for hub (e.g., 'username/socrates-instruction-dataset'): ")
        print("Pushing to hub...")
        final_dataset.push_to_hub(
            hub_name,
            token=os.getenv("HUGGINGFACE_TOKEN"),
        )
    
    print("Done!")
    
    # Print some examples
    print("\nSample conversations:")
    for i, example in enumerate(final_dataset[:3]):
        print(f"\nExample {i+1}:")
        for msg in example["conversations"]:
            print(f"{msg['from']}: {msg['value']}")

if __name__ == "__main__":
    main()