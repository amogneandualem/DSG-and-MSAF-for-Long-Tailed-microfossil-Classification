import os, glob, torch, random, time, logging, gc
import numpy as np
import cv2
from PIL import Image, ImageOps
from diffusers import FluxImg2ImgPipeline
from skimage import exposure

# ===================== A800 FINAL HQ CONFIG =====================
RESOLUTION = 1024    
STEPS = 35           # Increased for razor-sharp lattice pores
STRENGTH = 0.65      # Higher strength to "paint" over black areas with white silica
GUIDANCE = 8.5       # Strong enforcement of scientific textures
MIN_FLOOR = 500      
MAX_CEILING = 800    
MODEL_PATH = "/aifs/user/home/amogneandualem/models/FLUX"
TRAIN_DIR = "/aifs/user/home/amogneandualem/New_project/Split_dataset/train"
LOG_FILE = "/aifs/user/home/amogneandualem/New_project/generation_A800.log"
# =================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])

SCIENTIFIC_PROMPTS = {
    "Acanthodesmia_micropora": [
        "SEM: Acanthodesmia micropora, bright secondary electron lattice, hexagonal pores, silica spines, topographic contrast, black background",
        "A. micropora test, secondary electron imaging, glowing biogenic opal, sharp lattice vertices, high contrast grayscale SEM micrograph"
    ],
    "Actinomma_leptoderma_boreale": [
        "SEM: Actinomma leptoderma, triple-shelled spheres, bright opal-A silica, electron charging spine tips, crystalline lattice glow, vacuum black",
        "Actinomma fossil, secondary electron scan, sharp radial beams, translucent silica architecture, brilliant white on black background"
    ],
    "Antarctissa_denticulata-cyrindrica": [
        "SEM: Antarctissa denticulata-cylindrica, cylindrical test, bright secondary electron contrast, sharp pore rows, topographic detail, microfossil",
        "Antarctissa microfossil, high-voltage electron imaging, bright opal skeleton, razor-sharp lattice, secondary electron detector contrast"
    ],
    "Antarctissa_juvenile": [
        "SEM: Antarctissa juvenile, early ontogenetic stage, simple lattice formation, bright silica texture, 50μm scale, secondary electron imaging",
        "Young Antarctissa, developing silica test, growth bands, high contrast SEM, secondary electron charging, sharp micro-features, black background"
    ],
    "Antarctissa_longa-strelkovi": [
        "SEM: Antarctissa longa-strelkovi, elongated cylindrical form, longitudinal pore rows, bright silica lattice, topographic contrast, sharp edges",
        "A. longa-strelkovi, secondary electron scan, extended cephalic structure, biogenic opal, high contrast grayscale, sharp microfossil detail"
    ],
    "Botryocampe_antarctica": [
        "SEM: Botryocampe antarctica, botryoidal chambers, grape-like cluster, bright secondary electron contrast, sharp chamber pores, silica glow",
        "B. antarctica cluster, secondary electron imaging, interconnected globular chambers, high contrast SEM, bright silica on black background"
    ],
    "Botryocampe_inflatum-conithorax": [
        "SEM: Botryocampe inflatum-conithorax, inflated chambers, conical thorax, bright silica lattice, topographic contrast, sharp pore morphology",
        "B. inflatum-conithorax, secondary electron imaging, complex chamber test, bright opal-A texture, sharp microfossil features, black vacuum"
    ],
    "Ceratocytris_historicosus": [
        "SEM: Ceratocytris historicosus, three-bladed apical horn, lattice cephalis, bright secondary electron contrast, sharp pores, topographic detail",
        "C. historicosus test, secondary electron scan, elaborate apical structure, brilliant silica lattice, sharp edges, high contrast SEM"
    ],
    "Cycladophora_bicornis": [
        "SEM: Cycladophora bicornis, double-horned morphology, bright secondary electron contrast, sharp thorax pores, topographic shadows, silica glow",
        "C. bicornis fossil, secondary electron imaging, distinct apical spines, biogenic opal structure, bright white on black background, 8k SEM"
    ],
    "Cycladophora_cornutoides": [
        "SEM: Cycladophora cornutoides, prominent apical horn, segmented test, bright secondary electron contrast, sharp pores, topographic detail",
        "C. cornutoides microfossil, secondary electron scan, biogenic silica architecture, razor-sharp lattice, high contrast grayscale SEM"
    ],
    "Cycladophora_davisiana": [
        "SEM: Cycladophora davisiana, cold-water morphotype, bright secondary electron charging, sharp pore patterns, brilliant silica, black background",
        "C. davisiana specimen, high contrast secondary electron imaging, sharp cephalic structure, biogenic opal lattice, sharp topographic shadows"
    ],
    "Diatoms": [
        "SEM: Centric diatom valve, radial areolae lattice, bright silica frustule, secondary electron edge-glow, sharp topographic contrast",
        "Diatom frustule, secondary electron detector, brilliant white silica texture, deep black background, sharp pores, linking spines, SEM"
    ],
    "Druppatractus_irregularis-bensoni": [
        "SEM: Druppatractus irregularis-bensoni, irregular double shell, bright secondary electron contrast, connecting beams, sharp pores, silica glow",
        "D. irregularis-bensoni, secondary electron scan, peanut-shaped test, biogenic opal texture, high contrast SEM, sharp lattice detail"
    ],
    "Eucyrtidium_spp": [
        "SEM: Eucyrtidium spp, multi-segmented conical test, bright secondary electron contrast, sharp pore patterns, topographic detail, silica lattice",
        "Eucyrtidium specimen, secondary electron imaging, evolutionary variation, brilliant white silica architecture, sharp micro-features, SEM"
    ],
    "Fragments": [
        "SEM: Radiolarian fragments, broken lattice structure, sharp fracture edges, bright secondary electron contrast, topographic detail, silica shards",
        "Microfossil fragments, secondary electron scan, weathering patterns, biogenic opal fragments, high contrast SEM, brilliant white on black"
    ],
    "Larcoids_inner": [
        "SEM: Larcoid inner structure, 3D lattice framework, bright secondary electron contrast, sharp internal beams, topographic detail, silica skeleton",
        "Larcoid framework, secondary electron imaging, internal architectural pores, brilliant white silica, sharp micro-features, SEM cross-section"
    ],
    "Lithocampe_furcaspiculata": [
        "SEM: Lithocampe furcaspiculata, forked terminal spines, segmented test, bright secondary electron contrast, sharp pores, topographic detail",
        "L. furcaspiculata, secondary electron scan, bifurcating spines, biogenic opal lattice, high contrast SEM, brilliant white on black background"
    ],
    "Lithocampe_platycephala": [
        "SEM: Lithocampe platycephala, flattened cephalis, bright secondary electron contrast, sharp lattice pores, topographic detail, silica glow",
        "L. platycephala test, secondary electron imaging, broad cephalic structure, brilliant white silica architecture, sharp microfossil features"
    ],
    "Lithomelissa_setosa-borealis": [
        "SEM: Lithomelissa setosa-borealis, setose surface, dense spines, bright secondary electron contrast, sharp pore patterns, topographic detail",
        "L. setosa-borealis, secondary electron scan, bristly surface texture, biogenic opal spines, high contrast SEM, brilliant white silica"
    ],
    "Lophophana_spp": [
        "SEM: Lophophana spp, apical horn, lattice cephalis, bright secondary electron contrast, sharp pore patterns, topographic detail, silica glow",
        "Lophophana specimen, secondary electron imaging, biogenic silica architecture, razor-sharp lattice, high contrast grayscale SEM micrograph"
    ],
    "Other_Nassellaria": [
        "SEM: Nassellarian radiolaria, conical test, bright secondary electron contrast, sharp lattice pores, topographic detail, silica architecture",
        "Nassellaria morphotype, secondary electron scan, biogenic opal structure, brilliant white on black background, high contrast SEM imaging"
    ],
    "Other_Spumellaria": [
        "SEM: Spumellarian radiolaria, spherical lattice, bright secondary electron contrast, sharp pores, topographic detail, radial symmetry, silica glow",
        "Spumellaria specimen, secondary electron imaging, biogenic opal architecture, razor-sharp lattice, high contrast grayscale SEM micrograph"
    ],
    "Phormospyris_stabilis_antarctica": [
        "SEM: Phormospyris stabilis antarctica, tripod structure, bright secondary electron contrast, hexagonal lattice pores, topographic detail",
        "P. stabilis antarctica, secondary electron scan, triradiate symmetry, brilliant white silica lattice, sharp edges, high contrast SEM"
    ],
    "Phortycium_clevei-pylonium": [
        "SEM: Phortycium clevei-pylonium, gated chambers, pylome opening, bright secondary electron contrast, sharp lattice pores, topographic detail",
        "P. clevei-pylonium, secondary electron scan, complex multi-gated test, brilliant white silica, sharp micro-features, SEM imaging"
    ],
    "Plectacantha_oikiskos": [
        "SEM: Plectacantha oikiskos, basket structure, radial spines, bright secondary electron contrast, sharp lattice connections, topographic detail",
        "P. oikiskos fossil, secondary electron imaging, complex 3D lattice, biogenic opal spines, bright white on black background, high contrast SEM"
    ],
    "Pseudodictyophimus_gracilipes": [
        "SEM: Pseudodictyophimus gracilipes, slender basal feet, conical test, bright secondary electron contrast, sharp lattice pores, topographic detail",
        "P. gracilipes test, secondary electron scan, biogenic silica architecture, razor-sharp lattice edges, high contrast grayscale SEM"
    ],
    "Sethoconus_tablatus": [
        "SEM: Sethoconus tablatus, tabulate segments, flat chambers, bright secondary electron contrast, sharp pore patterns, topographic detail",
        "S. tablatus fossil, secondary electron imaging, segmented test morphology, brilliant white silica, sharp micro-features, SEM background"
    ],
    "Siphocampe_arachnea_group": [
        "SEM: Siphocampe arachnea, spiderweb lattice, delicate mesh, bright secondary electron contrast, sharp pore connections, topographic detail",
        "S. arachnea group, secondary electron scan, web-like silica architecture, high contrast SEM, brilliant white on black background"
    ],
    "Spongodiscus": [
        "SEM: Spongodiscus, discoidal spongy mesh, bright secondary electron contrast, irregular silica network, topographic detail, spongy texture",
        "Spongodiscus specimen, secondary electron imaging, spongy biogenic opal, brilliant white on black background, high contrast SEM"
    ],
    "Spongurus_pylomaticus": [
        "SEM: Spongurus pylomaticus, cylindrical spongy test, pylomes, bright secondary electron contrast, sharp spongy mesh, topographic detail",
        "S. pylomaticus, secondary electron scan, cylindrical morphology, biogenic silica architecture, high contrast SEM, brilliant white on black"
    ],
    "Sylodictya_spp": [
        "SEM: Sylodictya spp, lace-like network, bright secondary electron contrast, regular geometric lattice, sharp pores, topographic detail",
        "Sylodictya specimen, secondary electron imaging, biogenic silica network, razor-sharp lattice edges, high contrast grayscale SEM"
    ],
    "Zygocircus": [
        "SEM: Zygocircus, basket arched test, bright secondary electron contrast, sharp connecting bars, topographic detail, circular lattice glow",
        "Zygocircus specimen, secondary electron scan, arched skeletal elements, brilliant white silica, sharp micro-features, SEM background"
    ]
}
def enhance_scientific_detail(image):
    """Scientific-grade post-processing to reveal hidden textures."""
    img_array = np.array(image.convert("L"))
    
    # 1. CLAHE: Brings out the lattice texture hidden in dark pixels
    enhanced = exposure.equalize_adapthist(img_array, clip_limit=0.03)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    # 2. Laplacian Sharpening: Makes the pore edges crisp for CNN features
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return Image.fromarray(sharpened)

def apply_canny_precharge(image):
    """Locks the fossil structure using edge detection."""
    img_array = np.array(image.convert("L"))
    edges = cv2.Canny(img_array, 50, 150)
    blended = cv2.addWeighted(img_array, 0.6, edges, 0.4, 0)
    return Image.fromarray(blended).convert("RGB")

def main():
    logging.info(">>> LOADING FLUX ULTRA-HQ PIPELINE...")
    pipe = FluxImg2ImgPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")

    classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    
    for class_name in classes:
        class_path = os.path.join(TRAIN_DIR, class_name)
        
        # 1. Count existing work
        hybrid_gens = glob.glob(os.path.join(class_path, "flux_hybrid_new*"))
        originals = [f for f in glob.glob(os.path.join(class_path, "*.*")) if "flux_" not in f]
        
        # 2. Cleanup only non-hybrid/non-labeled files
        old_junk = [f for f in glob.glob(os.path.join(class_path, "flux_*")) if "flux_hybrid_new" not in f]
        for f in old_junk: os.remove(f)

        target = max(MIN_FLOOR, min(MAX_CEILING, len(originals) * 4))
        gap = target - (len(originals) + len(hybrid_gens))
        
        if gap <= 0:
            logging.info(f"DONE: {class_name} has {len(originals)+len(hybrid_gens)} images.")
            continue

        logging.info(f"*** {class_name}: Generating {gap} Ultra-HQ images to reach {target} ***")

        for i in range(gap):
            try:
                base_img_path = random.choice(originals)
                raw_img = Image.open(base_img_path).convert("RGB").resize((RESOLUTION, RESOLUTION))
                init_image = apply_canny_precharge(raw_img)
                
                # REFINED PROMPT INJECTION
                visual_header = "ULTRA-HIGH CONTRAST SEM, SECONDARY ELECTRON SCAN, GLOWING WHITE SILICA LATTICE, "
                species_prompt = random.choice(SCIENTIFIC_PROMPTS.get(class_name, [f"Microfossil specimen of {class_name}, sharp pores"]))
                prompt = visual_header + species_prompt

                output = pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=STRENGTH,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                ).images[0]

                # APPLY SCIENTIFIC ENHANCEMENT (CLAHE + Sharp)
                final_img = enhance_scientific_detail(output)
                
                save_name = f"flux_hybrid_new_{int(time.time())}_{i}.jpg"
                final_img.save(os.path.join(class_path, save_name), quality=95)
                
                # Cleanup VRAM
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logging.error(f"Error in {class_name}: {e}")
                time.sleep(1)
                continue

    logging.info(">>> ULTRA-HQ DATASET COMPLETE.")

if __name__ == "__main__":
    main()