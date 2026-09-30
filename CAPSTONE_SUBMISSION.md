# Capstone Submission Checklist

Repository:
https://github.com/aditee-dhondge/CUDA-at-Scale-Independent-Project

## Materials

- README.md with project purpose, algorithm, CUDA mapping, memory operations, CLI usage, validation, results, challenges, lessons learned, and future work.
- src/batch_image_blur.cu with the CUDA implementation.
- Makefile for compilation.
- run.sh for reproducible dataset generation, build, execution, and verification.
- verify_outputs.py for checking all generated PGM outputs.
- proof_of_execution.txt containing documented NVIDIA L4 execution evidence.
- docs/project_description.md with detailed project explanation.
- docs/presentation_script.md with a 5-10 minute presentation script.

## Rubric Coverage

### Code Repository
The repository contains the CUDA source, descriptive README, command-line arguments, Makefile, run script, and output-validation support.

### Proof of Execution
The execution proof documents a complete 200-image NVIDIA L4 run with 13,107,200 processed pixels and 200 generated outputs.

### Project Description
The project description explains the purpose, algorithm, CUDA kernel mapping, memory operations, timing methodology, batch validation, challenges, lessons learned, results, and future work.

### Presentation
A presentation script and deck have been prepared. The remaining platform-side requirement is to provide a public URL to the recorded 5-10 minute presentation.

## Submission
Submit the public repository URL, proof artifact, detailed project description, and public presentation URL through the Coursera peer-graded assignment interface.
