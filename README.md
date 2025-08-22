# alt-text-generation
This repository contains data and training scripts for fine-tuning **LLaVA-7B** to generate image descriptions tailored for blind and low-vision (BLV) users. The model is fine-tuned on preference data (alternative text vs. captions) using **LoRA** for parameter-efficient adaptation and **DPO** for alignment with BLV users’ preferences.

**Accessibilité visuelle et éducation inclusive: Étude préliminaire sur la génération de textes alternatifs**  
[Link to paper](https://ia-edu.lis-lab.fr/doc/articles/IA-%C3%89DU_2025_paper_205.pdf)

<p align="center">
  <img src="architecture.png" alt="Training pipeline diagram" width="60%">
</p>
