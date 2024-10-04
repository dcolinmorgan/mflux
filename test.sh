mflux-generate-controlnet \
  --prompt "A beautiful fashion model" \
  --model schnell \
  --steps 5 \
  --seed 1727047657 \
  --height 1066 \
  --width 692 \
  -q 8 \
  --lora-paths "/Users/apple/Downloads/FashionPhotographyStylev2Epoch10.safetensors" \
  --controlnet-image-path "/Users/apple/WRK/ComfyUI/input/orig/p09.jpg" \
  --controlnet-strength 0.5 \
  --control-mode 4

# pip install .
# bash test.sh
