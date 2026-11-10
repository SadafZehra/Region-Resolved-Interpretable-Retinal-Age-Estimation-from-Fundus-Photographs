# Where Does the Retina Show Its Age?
### Interpretable retinal age prediction and the "retinal age gap" as a marker of systemic health

**Data:** ODIR-5K (Ocular Disease Intelligent Recognition)

---

## 1. Background and questions

Deep networks can estimate a person's age from a colour fundus photograph. The difference between predicted and real age (the **retinal age gap**) has been linked in large cohorts to systemic conditions such as diabetes, hypertension and cardiovascular risk. Most of these models are black boxes, so a clinician cannot see *which* part of the retina makes someone "look older".

1. How well can a ResNet-50 predict age from **healthy** fundus images?
2. Is the age gap larger in eyes with signs of disease (diabetic and hypertensive retinopathy, and others) than in healthy eyes, after correcting for regression to the mean, age and sex? Does it grow with DR severity?
3. **Where** does the model read age from? Two models are compared:
   - **ResNet-50 + Grad-CAM** (post-hoc explanation)
   - **Evidence-map regressor** (inherently interpretable): a 1×1 conv gives an "age map", and the predicted age is exactly `mean_age + std × spatial mean(map)`, so every region's contribution can be read **in years**.

   Attribution is split into automatically found anatomical regions: optic disc, macula, vessels, periphery, other retina and background.
4. Does the reading pattern change between young and old people (population-average maps aligned by optic-disc side)?
5. Are the explanations faithful (deletion test)?

## 2. Method

- **Data.** ODIR-5K via `kagglehub` (public): about 5,000 patients from hospitals in China, with age, sex, left and right colour fundus photos from several cameras and doctors' diagnostic keywords per eye. Please follow the dataset's terms and cite it.
- **Quality filtering.** Eyes with `low image quality`, `lens dust`, `no fundus image`, etc. are removed. 6,426 usable eyes remain (from 6,716).
- **Training population.** Age models are trained **only on healthy patients** (both eyes "normal fundus", 999 patients), with a **patient-level split** (stratified by age): 649 train / 150 validation / 200 test patients (1,298 / 300 / 400 eyes). Ages 18 to 90; train age 56.8 ± 11.1 years.
- **Disease eyes** (3,012 eyes) are used only for the age-gap analysis and only from patients never used in training or validation. Groups: diabetic retinopathy, hypertensive retinopathy, glaucoma, AMD, drusen, cataract, myopia.
- **Models.** ResNet-50 (ImageNet weights), 384 px, dilated last stage (24×24 feature maps), mixed precision, mild colour augmentation only (lens tint carries age information), cosine LR, early stopping on validation MAE.
- **Bias-corrected age gap.** Regression to the mean is removed using a linear correction fitted on the validation set. Group differences are tested with Mann-Whitney and with a model adjusted for age and sex; DR severity is related to the gap with Spearman correlation.
- **Anatomy masks** are found automatically (brightest blob = optic disc, darkest spot = macula, Frangi filter = vessels, periphery, background).
- **Faithfulness.** Deletion test: blurring the most-attributed patches vs. random patches.

## 3. Results (from the saved run)

### 3.1 Age prediction on healthy test patients (400 eyes, 200 patients)

| Model | MAE (years) | 95 % CI | Pearson r | R² |
|---|---|---|---|---|
| ResNet-50 + Grad-CAM | 6.81 | 6.27 to 7.32 | 0.594 | 0.352 |
| Evidence-map regressor | 6.83 | 6.25 to 7.38 | 0.591 | 0.338 |

Both models perform the same. The interpretable model costs nothing in accuracy. Published models trained on very large cohorts report MAEs of roughly 3 to 4 years; the higher error here is expected with a small training set (1,298 eyes). Both models show the usual regression to the mean (bias slope about −0.6).

### 3.2 Corrected retinal age gap vs. healthy patients (adjusted for age and sex)

Adjusted difference in years (positive = looks older than healthy eyes of the same age and sex):

| Group | Evidence model | Grad-CAM model |
|---|---|---|
| Cataract | +4.44 (p < 0.001) | +4.11 (p < 0.001) |
| AMD | +2.70 (p < 0.001) | +3.35 (p < 0.001) |
| Drusen | +1.91 (p = 0.004) | +2.51 (p < 0.001) |
| Glaucoma | +1.86 (p = 0.002) | +2.32 (p < 0.001) |
| Hypertensive retinopathy | −0.01 (p = 0.99) | +1.18 (p = 0.058) |
| Diabetic retinopathy | +0.39 (p = 0.40) | see notebook output |
| Myopia | −5.13 (p < 0.001) | see notebook output |

**Dose-response with DR severity** (Spearman rho, mean gap per grade 0 to 4):
- Grad-CAM model: rho = 0.20; mean gap −1.12, −1.87, +0.42, +1.43, **+4.65** years (healthy to PDR)
- Evidence model: rho = 0.12; mean gap −0.98, −2.06, −0.03, +0.21, **+4.29** years

### 3.3 Where does the model read age from?

Median "lift" per region (share of attribution divided by share of area; above 1 = looks there more than its size):

| Model | Age group | Optic disc | Macula | Vessels | Periphery | Other retina | Background |
|---|---|---|---|---|---|---|---|
| Evidence | < 45 | 3.44 | 2.98 | 1.39 | 0.59 | 1.32 | 0.13 |
| Evidence | 45 to 60 | 4.02 | 2.07 | 1.35 | 0.67 | 1.26 | 0.16 |
| Evidence | > 60 | 3.34 | 1.43 | 1.39 | 0.68 | 1.37 | 0.16 |
| Grad-CAM | < 45 | 1.91 | 2.62 | 1.39 | 0.70 | 1.29 | 0.13 |
| Grad-CAM | 45 to 60 | 1.63 | 1.49 | 1.36 | 0.89 | 1.30 | 0.17 |
| Grad-CAM | > 60 | 1.87 | 1.33 | 1.31 | 0.93 | 1.33 | 0.20 |

Both models focus on the **optic disc, macula and vessels** and largely ignore the periphery and the black background. For younger people (< 45) both models weight the **macula** more strongly than for older people (evidence model: 2.98 vs. 1.43), so the reading pattern does change with age.

**Region contributions in years (evidence model)** are small for every region (about −0.7 to +0.6 years vs. the training mean), and the differences between hypertensive or diabetic eyes and healthy eyes are small (about 0.6 years or less). Only a few differences reach p < 0.05 (hypertensive: periphery +0.19, other retina −0.62, background +0.06), so no single region clearly explains an older-looking retina in these groups.

### 3.4 Faithfulness (deletion test, mean absolute change in predicted age in years)

| Model | 5 % patches blurred: top vs. random | 10 % | 20 % |
|---|---|---|---|
| Evidence | 4.12 vs. 1.22 | 4.83 vs. 2.13 | 6.09 vs. 3.59 |
| Grad-CAM | 2.25 vs. 0.62 | 2.98 vs. 0.96 | 3.90 vs. 1.89 |

Removing the most-attributed patches changes the prediction clearly more than removing random patches for both models, so the explanations are faithful. The gap between top and random is larger for the evidence model.

## 4. Interpretation

- A ResNet-50 trained on about 650 healthy patients predicts age with an MAE of about 6.8 years and r ≈ 0.59, as a pilot-level result.
- Eyes with **cataract, AMD, drusen and glaucoma look 2 to 4 years older** than healthy eyes. Cataract is mostly a sanity check: haze and lens changes alter image appearance, so part of that gap is image quality and not biology.
- **Diabetic and hypertensive retinopathy do not show a clear overall gap** with the evidence model, but the **gap grows with DR severity** (about +4 to +5 years for proliferative DR), a dose-response pattern. This pattern is stronger evidence than a single group difference, but it may reflect lesions the model never saw during healthy-only training.
- **Myopia looks about 5 years younger** with the evidence model, probably an image or age-distribution effect (myopic patients in ODIR tend to be younger, and fundus appearance differs) rather than a biological finding.
- The interpretable model matches the black box in accuracy and in faithfulness, and gives region contributions directly in years.

## 5. Limitations

- ODIR "hypertensive retinopathy" and "diabetic retinopathy" are **eye findings**, not confirmed systemic diagnoses or blood pressure / HbA1c values. A full study needs cohort data with systemic measurements (for example UK Biobank or the German National Cohort).
- Disease eyes contain lesions the model never saw in training (healthy-only training), so part of the gap may be distribution shift. Region-level contributions help separate "older-looking vessels" from "lesions confuse the model", but the differences found were small.
- The anatomy masks are automatic and approximate.
- Single seed, single dataset (one country), and a small training set (649 patients). Repeat with 3 seeds and an external dataset before drawing conclusions.
- Group sizes differ a lot (for example 103 hypertensive vs. 1,096 DR patients), so confidence is not equal across groups.

