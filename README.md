# Automated Story Generation Using GPT-2

This project utilizes the GPT-2 language model to generate coherent and engaging stories based on given prompts. By leveraging the power of GPT-2, the model can produce human-like text that can be used for creative writing, storyboarding, and more.

## Introduction

GPT-2 (Generative Pre-trained Transformer 2) is a large transformer-based language model developed by OpenAI. It can generate coherent and contextually relevant text given an initial prompt. This project uses GPT-2 to generate stories by providing a starting sentence or a brief description.

## Prerequisites

- Python 3.7 or higher
- PyTorch
- Transformers

You can install the required packages using the following command:

```bash
pip install torch transformers
```

## Project Structure

```
Story-Generator/
│
├── models/
│   └── gpt2_story_generator.pth  # Saved model checkpoint (optional)
│
├── generate_story.py     # Script to generate stories
│
└── README.md             # Project README file
```

### Generating Stories

To generate stories, run the `generate_story.py` script. You can specify the prompt to start the story and other generation parameters such as length and temperature.


## Contributing

Contributions are welcome! If you have any ideas or improvements, feel free to fork the repository and submit a pull request.


Feel free to modify this README to suit your specific project details and requirements.
