# Digital Image Processing Visualiser

**Live Demo:** [https://rushika1256.github.io/dip-visualiser/](https://rushika1256.github.io/dip-visualiser/)

An interactive, 3D web application built with Three.js that visualizes complex concepts from Rafael C. Gonzalez's *Digital Image Processing* textbook. This tool helps you learn and explore the math behind how computers see, process, and compress images.

## What it does

This project brings algorithms from the book to life by mapping them into 3D space:

### Chapter 7: Multiresolution & Wavelets
* **Wavelet Subbands (DWT):** Mathematically breaks an image down into its edges and approximations (using Haar and Daubechies-4 algorithms) and pulls them apart in 3D. 
* **Inverse Wavelet Transforms:** Watch how the image gets rebuilt data. You can delete pieces of the image data (thresholding) to see how lossy compression works.
* **Image Pyramids:** Visualizes Gaussian (blurred downsampling) and Laplacian (difference) pyramids stacked on top of each other.

### Chapter 8: Image Compression
* **Huffman Coding:** Analyzes the image to show an intensity histogram, tracks how much compression you get, and shows you the actual binary dictionary tree used to shrink the image.
* **Run-Length Encoding (RLE):** Converts the image to black-and-white, then groups repeating pixels together to see how much smaller the file gets.
* **Wavelet Lossy Compression (JPEG2000 style):** A pipeline that throws away weak textures, compresses the image, then reconstructs it while calculating quality metrics like PSNR.

## How to run it locally

1. Make sure you have [Node.js](https://nodejs.org/) installed.
2. Clone this repository:
   ```bash
   git clone https://github.com/rushika1256/dip-visualiser.git
   cd dip-visualiser
   ```
3. Install the dependencies:
   ```bash
   npm install
   ```
4. Start the local server:
   ```bash
   npm run dev
   ```
5. Open your browser and go to `http://localhost:5173/`

## Tech Stack
* **Three.js** (WebGL 3D Rendering)
* **Vite** (Build Tool)
* **Vanilla HTML/CSS/JS** (No heavy frameworks)
* **GSAP** (Smooth Animations)
