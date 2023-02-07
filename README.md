# CPL-OT
 
Entity alignment aims to discover unique equivalent entity pairs with the same meaning across different knowledge graphs (KG). It has been a compelling but challenging task for knowledge integration or fusion. Existing models have primarily focused on projecting KGs into a latent embedding space to capture inherent semantics between entities for entity alignment. However, the adverse impacts of alignment conflicts have been largely overlooked during training, thus limiting the entity alignment performance. To address this issue, we propose a novel Conflict-aware Pseudo Labeling via Optimal Transport model (CPL-OT) for entity alignment. The key idea of CPL-OT is to iteratively pseudo-label alignment pairs empowered with conflict-aware Optimal Transport modeling to boost the precision of entity alignment. CPL-OT is composed of two key components-entity embedding learning with global-local aggregation and iterative conflict-aware pseudo labeling-that mutually reinforce each other. To mitigate alignment conflicts during pseudo labeling, we propose to use optimal transport (OT) as an effective means to warrant one-to-one entity alignment between two KGs with the minimal overall transport cost. The transport cost is calculated as the rectified distance between entity embeddings obtained via graph convolution augmented with global-level semantics. Extensive experiments on benchmark datasets show that CPL-OT can markedly outperform state-of-the-art baselines under both settings with and without prior alignment seeds.

![CPL-OT](https://user-images.githubusercontent.com/93415770/173173155-9ff28371-06ec-4f92-aeb0-c011cec46432.jpg)

Environment:\
python--3.7.6;\
pytorch--1.7.1;\
cudatoolkit--10.1.243.

Source codes instructions:\
1.Download the "data.zip" from https://drive.google.com/file/d/13bPEknk0BVzwSzXL_cgK7PraKuR-vKfV/view?usp=share_link, then extract it to replace the original "data" folder in the depository;\
2.Open the file named "training_dbp15k.py" and "training_srprs.py" with Spyder to run CPL-OT on DBP15K and SRPRS respectively;\
3.Click "Run file".
