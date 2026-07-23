from me_mkm import BepInteraction, InitialStateInteraction, MEMKMBuilder, Reaction
from ..common import EMPTY, CO, O

def generate_model(beta, tile, delta_scale=1e-4, alpha=1.6, gamma=1e-3, kr=1.0,
                   khop_scale=1000.0, eps=8368.0, temperature=500.0):
    """ME-MKM generator at one beta.

    The physical parameters mirror the kMC model (co_oxidation.kmc.KMCParams)
    so both phases describe the same chemistry; the sweep drivers thread the
    same CLI values into both:

      alpha        CO adsorption rate, s^-1
      gamma        CO desorption prefactor, s^-1
      kr           CO + O reaction prefactor, s^-1
      khop_scale   fast-diffusion factor: khop = khop_scale * max(beta, alpha)
      eps          CO-CO nearest-neighbour repulsion, J/mol (enters as -eps)
      temperature  temperature, K
      delta_scale  O2 desorption scale: delta = delta_scale * beta; 0.0 for
                   irreversible O2 adsorption.
    """
    khop = khop_scale * max(beta, alpha)
    rates = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta_scale * beta,
        "kr": kr,
        "kh": khop,
    }
    # Eigenvalues and eigenvectors of the W, ask AI. 2D analog with observables a L.C of eigenvectors.
    # The more eligant solution is spectral theory.
    # Overcoming a heirarty of timescale seperations
    RT = 8.314462618 * temperature # J/mol
    interaction_matrix = [ # J/mol
        [0, 0,   0],
        [0, -eps , 0],
        [0, 0,   0]
    ]
    lateral = InitialStateInteraction(interaction_matrix, kbt=RT)
    co_hop  = BepInteraction(interaction_matrix, 0.5, kbt=RT)

    # Define the reactions
    reactions = [
        Reaction([EMPTY], [CO], rate=rates["alpha"], name="CO_ads", rate_symbol="α", rate_symbol_latex=r"\alpha"),
        Reaction([CO], [EMPTY], rate=rates["gamma"], name="CO_des", rate_symbol="γ", rate_symbol_latex=r"\gamma"),
        Reaction([EMPTY, EMPTY], [O, O], rate=rates["beta"], name="O2_ads", rate_symbol="β", rate_symbol_latex=r"\beta"),
        Reaction([O, O], [EMPTY, EMPTY], rate=rates["delta"], name="O2_des", rate_symbol="δ", rate_symbol_latex=r"\delta"),
        Reaction([CO, O], [EMPTY, EMPTY], rate=rates["kr"], name="CO_oxd", rate_symbol="kr", rate_symbol_latex=r"k_{r}"),
        Reaction([CO, EMPTY], [EMPTY, CO], rate=rates["kh"], name="CO_hop", rate_symbol="kh", rate_symbol_latex=r"k_{h}", interaction=co_hop),
        Reaction([O, EMPTY], [EMPTY, O], rate=rates["kh"], name="O_hop", rate_symbol="kh", rate_symbol_latex=r"k_{h}")
    ]

    species_names=["*", "CO", "O"]
    builder = MEMKMBuilder(tile, reactions, species_names, lateral)
    # graph_data = build_graph(builder)
    # save_html(graph_data, "co_oxidation.html")

    return builder
