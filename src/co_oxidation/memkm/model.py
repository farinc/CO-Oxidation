from me_mkm import BepBarrierModel, FrozenTransitionStateBarrier, MEMKMBuilder, Reaction
from ..common import EMPTY, CO, O

def generate_model(k_o_ads, tile, k_o_des_scale=1e-4, k_co_ads=1.6, k_co_des=1e-3,
                   k_rxn=1.0, khop_scale=1000.0, eps=8368.0, temperature=500.0):
    """ME-MKM generator at one k_o_ads.

    The physical parameters mirror the kMC model (co_oxidation.kmc.KMCParams)
    so both phases describe the same chemistry; the sweep drivers thread the
    same CLI values into both:

      k_co_ads     CO adsorption rate, s^-1
      k_co_des     CO desorption prefactor, s^-1
      k_rxn        CO + O reaction prefactor, s^-1
      khop_scale   fast-diffusion factor: khop = khop_scale * max(k_o_ads, k_co_ads)
      eps          CO-CO nearest-neighbour repulsion, J/mol (enters as -eps)
      temperature  temperature, K
      k_o_des_scale  O2 desorption scale: k_o_des = k_o_des_scale * k_o_ads; 0.0 for
                   irreversible O2 adsorption.
    """
    khop = khop_scale * max(k_o_ads, k_co_ads)
    rates = {
        "k_co_ads": k_co_ads,
        "k_o_ads": k_o_ads,
        "k_co_des": k_co_des,
        "k_o_des": k_o_des_scale * k_o_ads,
        "k_rxn": k_rxn,
        "khop": khop,
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
    lateral = FrozenTransitionStateBarrier(interaction_matrix, kbt=RT)
    co_hop  = BepBarrierModel(interaction_matrix, 0.5, kbt=RT)

    # Define the reactions
    reactions = [
        Reaction([EMPTY], [CO], rate=rates["k_co_ads"], name="CO_ads", rate_symbol="k_CO,ads", rate_symbol_latex=r"k_{\mathrm{CO,ads}}"),
        Reaction([CO], [EMPTY], rate=rates["k_co_des"], name="CO_des", rate_symbol="k_CO,des", rate_symbol_latex=r"k_{\mathrm{CO,des}}"),
        Reaction([EMPTY, EMPTY], [O, O], rate=rates["k_o_ads"], name="O2_ads", rate_symbol="k_O,ads", rate_symbol_latex=r"k_{\mathrm{O,ads}}"),
        Reaction([O, O], [EMPTY, EMPTY], rate=rates["k_o_des"], name="O2_des", rate_symbol="k_O,des", rate_symbol_latex=r"k_{\mathrm{O,des}}"),
        Reaction([CO, O], [EMPTY, EMPTY], rate=rates["k_rxn"], name="CO_oxd", rate_symbol="k_rxn", rate_symbol_latex=r"k_{\mathrm{rxn}}"),
        Reaction([CO, EMPTY], [EMPTY, CO], rate=rates["khop"], name="CO_hop", rate_symbol="k_hop", rate_symbol_latex=r"k_{\mathrm{hop}}", interaction=co_hop),
        Reaction([O, EMPTY], [EMPTY, O], rate=rates["khop"], name="O_hop", rate_symbol="k_hop", rate_symbol_latex=r"k_{\mathrm{hop}}")
    ]

    species_names=["*", "CO", "O"]
    builder = MEMKMBuilder(tile, reactions, species_names, lateral)
    # graph_data = build_graph(builder)
    # save_html(graph_data, "co_oxidation.html")

    return builder
