import os
import random

subjects = [
    "cat", "dog", "unicorn", "dragon", "robot", "astronaut", "knight", "butterfly",
    "mountain", "castle", "city", "forest", "ocean", "flower", "bird", "car", "street artist"
]

styles = [
    "photorealistic", "oil painting", "watercolor", "digital art", "sketch", "surreal",
    "cyberpunk", "fantasy", "vibrant", "minimalist", "abstract", "futuristic"
]

scenes = [
    "riding a unicorn on a rainbow", "in a bustling Parisian cafe", "at dusk",
    "floating islands", "breathing fire in a dark forest", "walking on Mars",
    "under the stars", "in a neon-lit city", "on a serene beach", "in a mystical forest"
]

def generate_prompts(num_prompts=100):
    prompts = []
    for _ in range(num_prompts):
        subject = random.choice(subjects)
        style = random.choice(styles)
        scene = random.choice(scenes)
        prompt = f"A {style} {subject} {scene}"
        prompts.append(prompt)
    return prompts

if __name__ == "__main__":
    # 指定绝对路径
    save_dir = r"D:\WOOd.W的大学\科研\xju大创\ID\Immuno-Diffusion\epigenetic_encoding"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    prompts = generate_prompts()
    for i, p in enumerate(prompts, 1):
        print(f"{i}: {p}")

    output_file = os.path.join(save_dir, "generated_prompts.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"提示词已保存到 {output_file}")