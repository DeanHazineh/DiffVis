<h1 align="center"> Grayscale to Hyperspectral at Any Resolution Using A Phase-Only Lens</h1>

<h2> This page is in development. Files will be added and modified. This is an early access. </h2>
<p align="center">
  <a href="https://deanhazineh.github.io/publications/DiffVis/diffvis_home.html">
    <img src="https://img.shields.io/badge/Project-Website-blue?style=for-the-badge" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2412.02798">
    <img src="https://img.shields.io/badge/ArXiv-Paper-red?style=for-the-badge" alt="ArXiv Paper">
  </a>
</p>

This work explores the inverse problem of reconstructing a hyperspectral image from a single, blurry grayscale measurement captured on a filterless photosensor. Although challenging, this task is possible since a diffractive lens can encode hyperspectral information via chromatic abberration. 
Previously, this task was largely unexplored and unsolved; however, we introduce a new model that is capable of reversing the forward measurement (shown below) to produce high-quality estimates. 

<p align="center">
  <strong style="font-size:20px;">The Forward Measurement Model</strong>
</p>
<p align="center">
  <img src="imgs/updated_forward_website.png" alt="forward_model" width="1000">
</p>

<p align="center">
  <strong style="font-size:20px;">Solving the Inverse Problem with guided, patch diffusion</strong>
</p>
<p align="center">
  <img src="imgs/updated_reverse_website.png" alt="forward_model" width="1000">
</p>

<p align="center">
  <img src="imgs/img_543_gif.gif" alt="Denoising Gif" width="1000">
</p>

