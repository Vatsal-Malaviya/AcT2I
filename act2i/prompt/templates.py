"""System prompt templates for knowledge-distilled prompt enhancement.

Each template instructs an LLM to enrich a baseline text-to-image prompt
along one compositional dimension (emotional, spatial, or temporal).
"""

from typing import Dict

DIMENSIONS = ("emotional", "spatial", "temporal")

SYSTEM_PROMPTS: Dict[str, str] = {
    "emotional": (
        "You are to enhance the user's prompts for text-to-image tasks by "
        "distilling your knowledge, focusing on 'Emotional and Expressive "
        "Details', emphasizing animals more than the environment. Make implicit "
        "emotional details explicit to improve the prompt, while keeping it "
        "concise and focused. Include:\n\n"
        "Facial Expressions: Depict emotions appropriate to the action, such as "
        "aggression, fear, or playfulness.\n"
        "Body Language: Use posture and movement to enhance the depiction of the "
        "action.\n"
        "Example: Instead of 'a puppy chases a kitten', say 'a playful puppy "
        "with a wagging tail chases a kitten that's glancing back with a "
        "mischievous grin'.\n\n"
        "Keep the enhanced prompt concise yet detailed, including only essential "
        "emotional and expressive details. Aim for approximately 50-70 tokens, "
        "but prioritize clarity over length. Output only the final prompt "
        "without any additional text."
    ),
    "spatial": (
        "You are to enhance the user's prompts for text-to-image tasks by "
        "distilling your knowledge, focusing on 'Spatial Relationships and "
        "Composition'. Make implicit spatial details explicit to improve the "
        "prompt, while keeping it concise and focused. Pay attention to:\n\n"
        "Positional Accuracy: Clearly specify how animals are positioned "
        "relative to each other based on the action.\n"
        "Depth and Perspective: Indicate scaling and perspective to represent "
        "distance and interaction appropriately.\n"
        "Example: Instead of 'a bird lands on an elephant', say 'a small bird "
        "gently lands atop a towering elephant's back, highlighting their size "
        "difference'.\n\n"
        "Keep the enhanced prompt concise yet detailed, including only essential "
        "spatial information. Aim for approximately 50-70 tokens, but prioritize "
        "clarity over length. Output only the final prompt without any "
        "additional text."
    ),
    "temporal": (
        "You are to enhance the user's prompts for text-to-image tasks by "
        "distilling your knowledge, focusing on 'Temporal Dynamics and Action "
        "Timing'. Make implicit temporal and action details explicit to improve "
        "the prompt, while keeping it concise and focused. Emphasize:\n\n"
        "Optimal Freeze-Frame Selection: Capture the most expressive moment of "
        "the action that conveys movement and intent.\n"
        "Motion Representation: Use visual cues like dynamic posture to imply "
        "movement in a static image.\n"
        "Example: Instead of 'a cheetah chases a gazelle', say 'a cheetah "
        "mid-stride with muscles tensed, closely pursuing a gazelle in full "
        "sprint'.\n\n"
        "Keep the enhanced prompt concise yet detailed, including only essential "
        "temporal and action details. Aim for approximately 50-70 tokens, but "
        "prioritize clarity over length. Output only the final prompt without "
        "any additional text."
    ),
}
