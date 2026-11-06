# Dataset layout and filenames

[Back to the quick start](../README.md#get-started)


The dataset archive is provided through the data availability statement in the accompanying paper. It contains 355 NumPy (`.npy`) files: chemical gases (270), solvent vapors (45), and microbial headspace (40).

Extract `TBDT_Dataset.zip` into the repository root. If extraction creates an extra outer folder, move `TBDT_Dataset/` beside `README.md`.


| Dataset | Training folder | Held-out folder |
| --- | --- | --- |
| Chemical gases | `TBDT_Dataset/0_chemical_gas/training/` | `TBDT_Dataset/0_chemical_gas/heldout_test/` |
| Solvent vapors | `TBDT_Dataset/1_solvent/training/` | `TBDT_Dataset/1_solvent/heldout_test/` |
| Microbial headspace | `TBDT_Dataset/2_bacteria/training/` | `TBDT_Dataset/2_bacteria/heldout_test/` |

Each file contains an array of shape `(256, raw_vg, raw_time)`. The first axis indexes the 256 devices; the last two axes correspond to gate-voltage and time indices. The supplied training and held-out folders contain different chips. Both folders are required when loading data. The chemical-gas dataset is shared by the dry and humidity configurations.

### Dataset filename convention

Filenames encode six fields separated by underscores:

```text
Set<number>_<chip number>_<analyte>_<concentration>_<humidity>_<cycle>.npy
```

For example, the chemical-gas file `Set1_1_ND_1_50_1.npy` means:

| Field | Example | Meaning |
| --- | --- | --- |
| Set number | `Set1` | Set identifier 1 |
| Chip number | `1` | Chip 1 |
| Analyte | `ND` | Nitrogen dioxide (NO₂) |
| Concentration | `1` | 1 ppm |
| Humidity | `50` | 50% relative humidity (RH) |
| Cycle | `1` | Measurement cycle 1 |

For a solvent-vapor example, `Set6_2_P_NQ_A_1.npy` means:

| Field | Example | Meaning |
| --- | --- | --- |
| Set number | `Set6` | Set identifier 6 |
| Chip number | `2` | Chip 2 |
| Analyte | `P` | C₃H₇OH |
| Concentration | `NQ` | Not quantified |
| Humidity | `A` | Ambient (uncontrolled) humidity |
| Cycle | `1` | Measurement cycle 1 |

Chemical-gas concentrations are expressed in ppm. For solvent vapors and microbial headspace, the gas-phase concentration was not quantified and is labeled `NQ`; this does not mean 0 ppm. The humidity code `A` indicates ambient room humidity without RH control, rather than a measured RH percentage. In the analyte field, `A` denotes acetone.

Solvent analyte codes are `A` = C₃H₆O, `E` = C₂H₅OH, `M` = CH₃OH, `P` = C₃H₇OH, and `T` = C₇H₈.

Microbial codes are `CA` = *Candida albicans*, `EC` = *Escherichia coli*, `PSA` = *Pseudomonas aeruginosa*, `SA` = *Staphylococcus aureus*, and `SS` = *Streptococcus sanguinis*.
