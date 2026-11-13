import openmdao.api as om


def user_floater_mooring_capex(platform_hull_mass, mooring_mass):
    """
    User-defined CAPEX model for floater + moorings.

    Parameters
    ----------
    platform_hull_mass : float
        Platform hull mass [kg].
    mooring_mass : float
        Total mooring-system mass [kg], including lines and anchors.

    Returns
    -------
    float
        CAPEX of floater + moorings.
    """
    #4 €/kg for mooring lines, correlazione marsfritto per floater

    capex = 2.14308834*platform_hull_mass + 7689597.24562343 + 4.0 * mooring_mass  
    return capex


class FloaterMooringCAPEX(om.ExplicitComponent):
    """
    Computes a user-defined floater + mooring CAPEX from platform hull mass
    and mooring mass.
    """

    def setup(self):
        self.add_input("platform_hull_mass", val=0.0, units="kg")
        self.add_input("mooring_mass", val=0.0, units="kg")

        self.add_output("capex_floater_moorings", val=0.0, units="USD")

        self.declare_partials(
            of="capex_floater_moorings",
            wrt=["platform_hull_mass", "mooring_mass"],
        )

    def compute_partials(self, inputs, partials):
        partials["capex_floater_moorings", "platform_hull_mass"] = 2.14308834
        partials["capex_floater_moorings", "mooring_mass"] = 4.0

    def compute(self, inputs, outputs):
        outputs["capex_floater_moorings"] = user_floater_mooring_capex(
            inputs["platform_hull_mass"],
            inputs["mooring_mass"],
        )
