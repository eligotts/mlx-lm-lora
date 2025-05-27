from mlx_lm.generate import generate
from mlx_lm.utils import load

from pathlib import Path
import argparse
import random
import json

def load_model(model_path):
    """Load MLX model and tokenizer"""
    try:
        model, tokenizer = load(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

def generate_user_message(model, tokenizer, args, topic, conversation_history=None):
    """Generate a user message for the given topic"""
    if not conversation_history:
        system_prompt = f"You are to adapt the role of a human user. {args.user_role}"
        user_prompt = f"You are struggling with this topic: '{topic}'. Ask a specific question or describe a concrete issue you're facing."
    else:
        system_prompt = f"You are a user continuing a conversation. {args.user_role}"
        last_assistant_msg = conversation_history[-1]['content']
        user_prompt = f"The assistant just said:\n\n{last_assistant_msg}\n\nNow ask a follow-up question or respond naturally to continue the conversation about {topic}."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=args.max_tokens
        )
        
        return response.strip()
    except Exception as e:
        raise SystemError(f"Error generating user message: {e}")

def generate_assistant_message(model, tokenizer, args, user_message, topic, is_final_turn=False):
    """Generate an assistant response to the user message"""
    system_prompt = args.system_prompt or f"You are {args.assistant_name}, {args.assistant_role}. Provide helpful, accurate, and detailed responses about MLX and machine learning topics."
    
    conclusion_instruction = " Please provide a concise summary or conclusion to wrap up this topic." if is_final_turn else ""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_message}{conclusion_instruction}"}
    ]
    
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=args.max_tokens
        )
        
        return response.strip()
    except Exception as e:
        raise SystemError(f"Error generating assistant message: {e}")

def generate_conversation(model, tokenizer, args, topic):
    """Generate a complete conversation for the given topic"""
    conversation = []
    num_turns = random.randint(1, args.max_turns)
    
    print(f"  Generating {num_turns} turns for topic: {topic}")
    
    for turn in range(num_turns):
        # Generate user message
        user_message = generate_user_message(
            model, tokenizer, args, topic, 
            conversation_history=conversation if turn > 0 else None
        )
        conversation.append({"role": "user", "content": user_message})
        
        # Generate assistant response
        is_final = (turn == num_turns - 1)
        assistant_message = generate_assistant_message(
            model, tokenizer, args, user_message, topic, is_final_turn=is_final
        )
        conversation.append({"role": "assistant", "content": assistant_message})
        
        if args.dry_run:
            print(f"    Turn {turn + 1}:")
            print(f"    User: {user_message[:100]}...")
            print(f"    Assistant: {assistant_message[:100]}...")
    
    return {
        "messages": conversation,
        "metadata": {
            "topic": topic,
            "num_turns": num_turns,
            "model_used": args.model
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MLX dataset using MLX models")
    parser.add_argument('--model', required=True, help='MLX model path or identifier')
    parser.add_argument('--output', default='mlx_synthetic_dataset.jsonl', help='Output file path')
    parser.add_argument('--num-convos', type=int, default=50, help='Number of conversations to generate')
    parser.add_argument('--max-turns', type=int, default=6, help='Maximum turns per conversation')
    parser.add_argument('--max-tokens', type=int, default=256, help='Maximum tokens per generation')
    parser.add_argument('--assistant-name', default='Josie', help='Assistant name')
    parser.add_argument('--assistant-role', 
                       default='an elite MLX assistant created by Gökdeniz Gülmez', 
                       help='Assistant role description')
    parser.add_argument('--user-role', 
                       default='a curious MLX developer asking for help', 
                       help='User role description')
    parser.add_argument('--system-prompt', default='', help='Custom system prompt for assistant')
    parser.add_argument('--dry-run', action='store_true', help='Print conversations instead of saving')
    parser.add_argument('--file-mode', choices=['overwrite', 'append'], default='overwrite',
                       help='Whether to overwrite existing file or append to it (default: overwrite')
    parser.add_argument('--topics', nargs='+', required=True, 
                       help="List of topics to generate conversations for")
    
    args = parser.parse_args()
    
    print(f"🚀 Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    
    print(f"📊 Generating {args.num_convos} conversations on topics: {args.topics}")
    
    dataset = []
    for i in range(args.num_convos):
        topic = args.topics[i % len(args.topics)]
        print(f"🧠 Generating conversation {i+1}/{args.num_convos}")
        
        try:
            conversation = generate_conversation(model, tokenizer, args, topic)
            dataset.append(conversation)
            
            if args.dry_run:
                print("=" * 80)
                print(json.dumps(conversation, indent=2))
                print("=" * 80)
                
        except Exception as e:
            print(f"❌ Error generating conversation for topic '{topic}': {e}")
            continue
    
    if not args.dry_run and dataset:
        print(f"💾 Saving {len(dataset)} conversations to {args.output}")
        
        # Determine file mode
        file_mode = 'a' if args.file_mode == 'append' else 'w'
        
        with open(args.output, file_mode) as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")
        
        action = "appended to" if args.file_mode == 'append' else "saved to"
        print(f"✅ Dataset generation complete! {len(dataset)} conversations {action} {args.output}")
    elif args.dry_run:
        print(f"🔍 Dry run complete. Generated {len(dataset)} conversations.")
    else:
        print("❌ No conversations were generated successfully.")

if __name__ == "__main__":
    main()