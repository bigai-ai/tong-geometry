# TongGeometry

TongGeometry is a research project focused on automated geometry problem solving using machine learning. It provides a framework for constructing geometric diagrams, inferring facts, and solving complex geometry problems. The project includes a core Python library `tonggeometry`, interactive web applications, and scripts for training and running machine learning models.

TongGeometry is the collection of our efforts to build a geometry prover that can both propose valid olympiad-level geometry problems and solve problems at the same level. TongGeometry's proposals have been considered at the ***National High School Math League (Beijing)*** and [***US Ersatz Math Olympiad***](http://web.evanchen.cc/usemo.html). The system also solves ***all problems in IMO-AG-30***, a IMO-level geometry problem benchmarked initially established by [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5). TongGeometry is also the foundation of a more advanced geometry reasoning system used in [Seed-Prover](https://arxiv.org/abs/2507.23726) that collectively reseached [Silver Medal](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025) performance in IMO 2025.

TongGeometry was developed concurrently and independently with AlphaGeometry, as can be seen from the completely different domain specific language used in the system. The initial tech report was released on [Arxiv](https://arxiv.org/abs/2412.10673), late in 2024.

Along the journey, the team would like to thank many people for their assistance in the project. In particular, the AoPS community that openly shares their solutions of existing problems, Patrik Bak for very thoughtful and detailed discussion for designing automatic rubrics and generating symmetric proposals, Evan Chen for assistance in problem evaluation, and members of Team China.

We would now pay back the community with our efforts.

## Features

- **Geometric Construction**: Programmatically define geometric figures using a set of actions and constructors.
- **Inference Engine**: Automatically deduce new geometric facts from a given diagram using a deductive database.
- **Interactive Web Apps**: Streamlit-based applications for visualizing geometric constructions, debugging proofs, and evaluating model performance.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bigai-ai/tong-geometry.git
    cd tonggeometry
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the `tonggeometry` package:**
    ```bash
    python setup.py develop
    ```

## Project Structure

```
/
├── app.py                  # Main Streamlit application for geometric construction and proof visualization
├── requirements.txt        # Python dependencies
├── setup.py                # Setup script for the tonggeometry package
├── tonggeometry/           # Core Python library for geometry problem solving
├── model/                  # Scripts for training and running ML models
│   ├── solve.py            # Main script for solving problems with a model
│   └── trainer.py          # Script for training models
├── scripts/                # Utility and helper scripts
└── launch.sh               # Script for launching distributed data generation
```


### Core Library (`tonggeometry`)

The `tonggeometry` library forms the core of the project. It can be used to programmatically:
-   Define geometric constructions (`tonggeometry.action`, `tonggeometry.constructor`).
-   Represent geometric diagrams (`tonggeometry.diagram`).
-   Perform logical inferences based on deductive database (`tonggeometry.inference_engine`).

### Training and Serving LLM (`model`)

The `model` folder contains the standard scripts for fine-tuning and serving a model dedicated to geometry auxiliary completions. `train.py` performs training with prepared data and `solve.py` is a naive script to solve a formatted problem using a local GPU device.

## Usage

### Interactive Applications

The project includes the main Streamlit application:

**`app.py`**: An interactive web GUI for constructing geometric diagrams, tracing fact dependencies, and generating proofs.

    To run the app:
    ```bash
    streamlit run app.py
    ```

The [online manual](https://docs.google.com/presentation/d/189nRtv4w5q1yobcQivPLvvnyunjmo-q5zZYbQkIm0xU/edit?usp=sharing) specifies how the app can be used.

### Data Generation

TongGeometry replies on a large amount of auto-generated geometry data. In the project, we rented 10k CPU cores from the Volcengine and ran the distributed data generation program for 30 days. If you want to run your own, feel free to checkout `launch.sh`. To benefit the entire research community, we would also like to openly share all our generated data [here]().

### Model Training

The model was trained using scripts in `model`, with resources managed by a Slurm cluster. Detailed multi-stage training pipeline can be found in our paper. The trained model checkpoints can be found [here]().

## Citation

If you use this project in your research, please consider citing it.

```bibtex
@article{zhang2024proposing,
  title={Proposing and solving olympiad geometry with guided tree search},
  author={Zhang, Chi and Song, Jiajun and Li, Siyu and Liang, Yitao and Ma, Yuxi and Wang, Wei and Zhu, Yixin and Zhu, Song-Chun},
  journal={arXiv preprint arXiv:2412.10673},
  year={2024}
}
```

## License

This project is licensed under the GNU GPLv3 License. See the `LICENSE` file for details.
