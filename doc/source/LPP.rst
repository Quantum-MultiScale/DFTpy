
Local Pseudopotentials (LPPs) for OF-DFT
========================================

Some background
---------------

Orbital-Free Density Functional Theory (OF-DFT) is an electronic structure method based solely on the electron density. In Kohn-Sham DFT pseudopotentials typically include a nonlocal part, which requires the Kohn-Sham orbitals to be evaluated. Due to the absence of Kohn-Sham orbitals in OF-DFT, only local pseudopotentials can be employed. 

Several local pseudopotentials have been developed and successfully applied to selected elements. A fairly exhaustive list:

- Bulk-derived Local Pseudopotentials (`BLPs <https://doi.org/10.1103/PhysRevB.69.125109>`_)

- High-quality Local Pseudopotentials (`HQLPP <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00101>`_)

- Optimized Effective Local Pseudopotentials (`OEPP <https://doi.org/10.1063/1.4944989>`_)

The `PRG <https://sites.rutgers.edu/prg>`_ recently derived a new set of local pseudopotentials for transition metal elements which is available below (PGBRV and PPSL as they are based on the `GBRV <https://www.physics.rutgers.edu/gbrv/>`_ and `PSL <https://dalcorso.github.io/pslibrary/>`_ ultrasoft pseudopotentials). The new pseudos can be constructed for any element by following this `tutorial <tutorials/jupyter/lpps.ipynb>`_. The formalism behind this new class of pseudopotentials is described `here <>`_.


Download LPPs
-------------

.. raw:: html

    <style>
    .periodic-table {
        display: grid;
        grid-template-columns: repeat(18, 40px);
        gap: 4px;
        margin-top: 20px;
    }

    .element {
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #FEB423;
        color: red;
        font-weight: bold;
        border-radius: 5px;
    }

    .element a {
        color: inherit;
        text-decoration: none;
    }

    .element a.inactive {
        color: white;
        pointer-events: none;
    }

    .element.blank {
        background: transparent;
        pointer-events: none;
    }
    </style>

    <select id="pseudopotentials">
        <option value="PGBRV02">PGBRV02</option>
        <option value="PPSL02">PPSL02</option>
        <option value="PGBRV10">PGBRV10</option>
        <option value="PPSL10">PPSL10</option>
        <option value="HQLPP_upf">HQLPP_upf</option>
        <option value="HQLPP_recpot">HQLPP_recpot</option>
        <option value="BLPS_lda">BLPS_lda</option>
        <option value="BLPS_gga">BLPS_gga</option>
        <option value="OEPP_recpot">EOPP_recpot</option>
        <option value="OEPP_upf">EOPP_upf</option>
        <option value="OEPP_cpi">EOPP_cpi</option>
    </select>

    <div class="periodic-table">
     <!-- Row 1 (1 element + 16 blanks + 1) -->
        <div class="element H"><a href="#" id="el-H">H</a></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element He"><a href="#" id="el-He">He</a></div>

        <!-- Row 2 -->
        <div class="element Li"><a href="#" id="el-Li">Li</a></div>
        <div class="element Be"><a href="#" id="el-Be">Be</a></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element blank"></div>
        <div class="element B"><a href="#" id="el-B">B</a></div>
        <div class="element C"><a href="#" id="el-C">C</a></div>
        <div class="element N"><a href="#" id="el-N">N</a></div>
        <div class="element O"><a href="#" id="el-O">O</a></div>
        <div class="element F"><a href="#" id="el-F">F</a></div>
        <div class="element Ne"><a href="#" id="el-Ne">Ne</a></div>

        <!-- Row 3 -->
        <div class="element Na"><a href="#" id="el-Na">Na</a></div>
        <div class="element Mg"><a href="#" id="el-Mg">Mg</a></div>
        <div class="element blank"></div><div class="element blank"></div><div class="element blank"></div><div class="element blank"></div><div class="element blank"></div>
        <div class="element blank"></div><div class="element blank"></div><div class="element blank"></div><div class="element blank"></div><div class="element blank"></div>
        <div class="element Al"><a href="#" id="el-Al">Al</a></div>
        <div class="element Si"><a href="#" id="el-Si">Si</a></div>
        <div class="element P"><a href="#" id="el-P">P</a></div>
        <div class="element S"><a href="#" id="el-S">S</a></div>
        <div class="element Cl"><a href="#" id="el-Cl">Cl</a></div>
        <div class="element Ar"><a href="#" id="el-Ar">Ar</a></div>

        <!-- Row 4 -->
        <div class="element K"><a href="#">K</a></div>
        <div class="element Ca"><a href="#">Ca</a></div>
        <div class="element Sc"><a href="#">Sc</a></div>
        <div class="element Ti"><a href="#">Ti</a></div>
        <div class="element V"><a href="#">V</a></div>
        <div class="element Cr"><a href="#">Cr</a></div>
        <div class="element Mn"><a href="#">Mn</a></div>
        <div class="element Fe"><a href="#">Fe</a></div>
        <div class="element Co"><a href="#">Co</a></div>
        <div class="element Ni"><a href="#">Ni</a></div>
        <div class="element Cu"><a href="#">Cu</a></div>
        <div class="element Zn"><a href="#">Zn</a></div>
        <div class="element Ga"><a href="#">Ga</a></div>
        <div class="element Ge"><a href="#">Ge</a></div>
        <div class="element As"><a href="#">As</a></div>
        <div class="element Se"><a href="#">Se</a></div>
        <div class="element Br"><a href="#">Br</a></div>
        <div class="element Kr"><a href="#">Kr</a></div>

        <!-- Row 5 -->
        <div class="element Rb"><a href="#">Rb</a></div>
        <div class="element Sr"><a href="#">Sr</a></div>
        <div class="element Y"><a href="#">Y</a></div>
        <div class="element Zr"><a href="#">Zr</a></div>
        <div class="element Nb"><a href="#">Nb</a></div>
        <div class="element Mo"><a href="#">Mo</a></div>
        <div class="element Tc"><a href="#">Tc</a></div>
        <div class="element Ru"><a href="#">Ru</a></div>
        <div class="element Rh"><a href="#">Rh</a></div>
        <div class="element Pd"><a href="#">Pd</a></div>
        <div class="element Ag"><a href="#">Ag</a></div>
        <div class="element Cd"><a href="#">Cd</a></div>
        <div class="element In"><a href="#">In</a></div>
        <div class="element Sn"><a href="#">Sn</a></div>
        <div class="element Sb"><a href="#">Sb</a></div>
        <div class="element Te"><a href="#">Te</a></div>
        <div class="element I"><a href="#">I</a></div>
        <div class="element Xe"><a href="#">Xe</a></div>

        <div class="element Cs"><a href="#">Cs</a></div>
        <div class="element Ba"><a href="#">Ba</a></div>
        <div class="element La"><a href="#">La</a></div>
        <div class="element Hf"><a href="#">Hf</a></div>
        <div class="element Ta"><a href="#">Ta</a></div>
        <div class="element W"><a href="#">W</a></div>
        <div class="element Re"><a href="#">Re</a></div>
        <div class="element Os"><a href="#">Os</a></div>
        <div class="element Ir"><a href="#">Ir</a></div>
        <div class="element Pt"><a href="#">Pt</a></div>
        <div class="element Au"><a href="#">Au</a></div>
        <div class="element Hg"><a href="#">Hg</a></div>
        <div class="element Tl"><a href="#">Tl</a></div>
        <div class="element Pb"><a href="#">Pb</a></div>
        <div class="element Bi"><a href="#">Bi</a></div>
        <div class="element Po"><a href="#">Po</a></div>
        <div class="element At"><a href="#">At</a></div>
        <div class="element Rn"><a href="#">Rn</a></div>

        <div class="element Fr"><a href="#">Fr</a></div>
        <div class="element Ra"><a href="#">Ra</a></div>
        <div class="element Ac"><a href="#">Ce</a></div>
        <div class="element Rf"><a href="#">Rf</a></div>
        <div class="element Db"><a href="#">Db</a></div>
        <div class="element Sg"><a href="#">Sg</a></div>
        <div class="element Bh"><a href="#">Bh</a></div>
        <div class="element Hs"><a href="#">Hs</a></div>
        <div class="element Mt"><a href="#">Mt</a></div>
        <div class="element Ds"><a href="#">Ds</a></div>
        <div class="element Rg"><a href="#">Rg</a></div>
        <div class="element Cn"><a href="#">Cn</a></div>
        <div class="element Nh"><a href="#">Nh</a></div>
        <div class="element Fl"><a href="#">Fl</a></div>
        <div class="element Mc"><a href="#">Mc</a></div>
        <div class="element Lv"><a href="#">Lv</a></div>
        <div class="element Ts"><a href="#">Ts</a></div>
        <div class="element Og"><a href="#">Og</a></div>
    </div>

    <script>
    const links = {
          PGBRV02: {
                Sc: "",
                Ti: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Ti_gbrv_new.psp8",
                V:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/V_gbrv_new.psp8",
                Cu: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Cu_gbrv_new.psp8",
                Zn: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Zn_gbrv_new.psp8",
                Y:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Y_gbrv_new.psp8",
                Zr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Zr_gbrv_new.psp8",
                Nb: "",
                Mo: "",
                Pd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Pd_gbrv_new.psp8",
                Ag: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Ag_gbrv_new.psp8",
                Hf: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Hf_gbrv_new.psp8",
                Ta: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Ta_gbrv_new.psp8",
                Pt: "",
                Au: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Au_gbrv_new.psp8",
                Co: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Co_gbrv_new.psp8",
                Cd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Cd_gbrv_new.psp8",
                Cr: "",
                Hg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Hg_gbrv_new.psp8",
                Ni: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Ni_gbrv_new.psp8",
                Re: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Re_gbrv_new.psp8",
                Rh: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Rh_gbrv_new.psp8",
                Tc: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Tc_gbrv_new.psp8",
                W:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/W_gbrv_new.psp8",
                Mn:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Mn_gbrv_new.psp8",
                Fe:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Fe_gbrv_new.psp8",
                Ni:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Ni_gbrv_new.psp8",
                Ru:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Ru_gbrv_new.psp8",
                Rh:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Rh_gbrv_new.psp8",
                Os:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Os_gbrv_new.psp8",
                Ir:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV0.2/Ir_gbrv_new.psp8",
            },
            PPSL02: {
                Sc: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Sc_psl_new.psp8",
                Ti: "",
                V:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/V_psl_new.psp8",
                Cu: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Cu_psl_new.psp8",
                Zn: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Zn_psl_new.psp8",
                Y:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Y_psl_new.psp8",
                Zr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Zr_psl_new.psp8",
                Nb: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Nb_psl_new.psp8",
                Mo: "",
                Pd: "",
                Ag: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Ag_psl_new.psp8",
                Hf: "",
                Ta: "",
                Pt: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Pt_psl_new.psp8",
                Au: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Au_psl_new.psp8",
                Co: "",
                Cd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Cd_psl_new.psp8",
                Cr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Cr_psl_new.psp8",
                Hg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Hg_psl_new.psp8",
                Ni: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Ni_psl_new.psp8",
                Re: "",
                Rh: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Rh_psl_new.psp8",
		        Tc: "",
                W:  "",
                Fe:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Fe_psl_new.psp8",
                Ru:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Ru_psl_new.psp8",
                Ir:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL0.2/Ir_psl_new.psp8",
                
            },
            PGBRV10: {
                Sc: "",
                Ti: "",
                V:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/V_gbrv_new.psp8",
                Cu: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Cu_gbrv_new.psp8",
                Zn: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Zn_gbrv_new.psp8",
                Y:  "",
                Zr: "",
                Nb: "",
                Mo: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Mo_gbrv_new.psp8",
                Pd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Pd_gbrv_new.psp8",
                Ag: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Ag_gbrv_new.psp8",
                Hf: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Hf_gbrv_new.psp8",
                Ta: "",
                Pt: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Pt_gbrv_new.psp8",
                Au: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Au_gbrv_new.psp8",    
                Co: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Co_gbrv_new.psp8",
                Cd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Cd_gbrv_new.psp8",
                Cr: "",
                Hg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Hg_gbrv_new.psp8",
                Ni: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Ni_gbrv_new.psp8",
                Re: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Re_gbrv_new.psp8",
                Rh: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Rh_gbrv_new.psp8",
                Tc: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Tc_gbrv_new.psp8",
                W:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/W_gbrv_new.psp8",
                Mn:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Mn_gbrv_new.psp8",
                Fe:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Fe_gbrv_new.psp8",
                Ni:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Ni_gbrv_new.psp8",
                Ru:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Ru_gbrv_new.psp8",
                Os:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PGBRV1.0/Os_gbrv_new.psp8",
            },
            PPSL10: {
                Sc: "",
                Ti: "",
                V:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/V_psl_new.psp8",
                Cu: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Cu_psl_new.psp8",
                Zn: "",
                Y:  "",
                Zr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Zr_psl_new.psp8",
                Nb: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Nb_psl_new.psp8",
                Mo: "",
                Pd: "",
                Ag: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Ag_psl_new.psp8",
                Hf: "",
                Ta: "",
                Pt: "",
                Au: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Au_psl_new.psp8",
                Tl: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Tl_psl_new.psp8",
                La: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/La_psl_new.psp8",
                Ac: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Ac_psl_new.psp8",
                Co: "",
                Cd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Cd_psl_new.psp8",
                Cr: "",
                Hg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Hg_psl_new.psp8",
                Ni: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Ni_psl_new.psp8",
                Re: "",
                Rh: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Rh_psl_new.psp8",
		        Tc: "",
                W:  "",
                Ni:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Ni_psl_new.psp8",
                Ir:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/NLPP/PPSL1.0/Ir_gbrv_new.psp8",
            },
            HQLPP_recpot:{
                Ag: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ag_lps.pbe.recpot",
                Al: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/al_lps.pbe.recpot",
                Au: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/au_lps.pbe.recpot",
                Ba: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ba_lps.pbe.recpot",
                Be: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/be_lps.pbe.recpot",
                Bi: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/bi_lps.pbe.recpot",
                Ca: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ca_lps.pbe.recpot",
                Cd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/cd_lps.pbe.recpot",
                Co: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/co_lps.pbe.recpot",
                Cr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/cr_lps.pbe.recpot",
                Cs: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/cs_lps.pbe.recpot",
                Cu: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/cu_lps.pbe.recpot",
                Fe: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/fe_lps.pbe.recpot",
                Ga: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ga_lps.pbe.recpot",
                Hg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/hg_lps.pbe.recpot",
                In: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/in_lps.pbe.recpot",
                Ir: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ir_lps.pbe.recpot",
                K:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/k_lps.pbe.recpot",
                Li: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/li_lps.pbe.recpot",
                Mn: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/mn_lps.pbe.recpot",
                Mg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/mg_lps.pbe.recpot",
                Mo: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/mo_lps.pbe.recpot",
                Na: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/na_lps.pbe.recpot",
                Nb: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/nb_lps.pbe.recpot",
                Ni: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ni_lps.pbe.recpot",
                Os: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/os_lps.pbe.recpot",
                Pb: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/pb_lps.pbe.recpot",
                Pd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/pd_lps.pbe.recpot",
                Pt: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/pt_lps.pbe.recpot",
                Re: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/re_lps.pbe.recpot",
                Rh: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/rh_lps.pbe.recpot",
                Ru: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ru_lps.pbe.recpot",
                Sc: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/sc_lps.pbe.recpot",
                Sr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/sr_lps.pbe.recpot",
                Ta: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ta_lps.pbe.recpot",
                Tc: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/tc_lps.pbe.recpot",
                Ti: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/ti_lps.pbe.recpot",
                Tl: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/tl_lps.pbe.recpot",
                V:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/v_lps.pbe.recpot",
                W:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/w_lps.pbe.recpot",
                Y:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/y_lps.pbe.recpot",
                Zn: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/zn_lps.pbe.recpot",
                Zr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/recpot/zr_lps.pbe.recpot",
            },
            HQLPP_upf:{
                Ag: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ag_lps.pbe.upf",
                Al: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/al_lps.pbe.upf",
                Au: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/au_lps.pbe.upf",
                Ba: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ba_lps.pbe.upf",
                Be: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/be_lps.pbe.upf",
                Bi: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/bi_lps.pbe.upf",
                Ca: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ca_lps.pbe.upf",
                Cd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/cd_lps.pbe.upf",
                Co: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/co_lps.pbe.upf",
                Cr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/cr_lps.pbe.upf",
                Cs: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/cs_lps.pbe.upf",
                Cu: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/cu_lps.pbe.upf",
                Fe: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/fe_lps.pbe.upf",
                Ga: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ga_lps.pbe.upf",
                Hg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/hg_lps.pbe.upf",
                In: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/in_lps.pbe.upf",
                Ir: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ir_lps.pbe.upf",
                K:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/k_lps.pbe.upf",
                Li: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/li_lps.pbe.upf",
                Mn: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/mn_lps.pbe.upf",
                Mg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/mg_lps.pbe.upf",
                Mo: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/mo_lps.pbe.upf",
                Na: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/na_lps.pbe.upf",
                Nb: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/nb_lps.pbe.upf",
                Ni: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ni_lps.pbe.upf",
                Os: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/os_lps.pbe.upf",
                Pb: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/pb_lps.pbe.upf",
                Pd: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/pd_lps.pbe.upf",
                Pt: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/pt_lps.pbe.upf",
                Re: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/re_lps.pbe.upf",
                Rh: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/rh_lps.pbe.upf",
                Ru: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ru_lps.pbe.upf",
                Sc: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/sc_lps.pbe.upf",
                Sr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/sr_lps.pbe.upf",
                Ta: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ta_lps.pbe.upf",
                Tc: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/tc_lps.pbe.upf",
                Ti: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/ti_lps.pbe.upf",
                Tl: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/tl_lps.pbe.upf",
                V:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/v_lps.pbe.upf",
                W:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/w_lps.pbe.upf",
                Y:  "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/y_lps.pbe.upf",
                Zn: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/zn_lps.pbe.upf",
                Zr: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/HQLPP/upf/zr_lps.pbe.upf",
            },
            BLPS_lda:{
                Ag: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//ag.lda.upf",
                Al: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//al.lda.upf",
                As: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//as.lda.upf",
                Ga: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//ga.lda.upf",
                In: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//in.lda.upf",
                Li: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//li.lda.upf",
                Mg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//mg.lda.upf",
                P: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//p.lda.upf",
                Sb: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//sb.lda.upf",
                Si: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//si.lda.upf",
            },

            BLPS_gga:{
                Al: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//al.gga.upf",
                As: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//as.gga.upf",
                Ga: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//ga.gga.upf",
                In: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//in.gga.upf",
                Li: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//li.gga.1.upf",
                Mg: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//mg.gga.upf",
                P: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//p.gga.upf",
                Sb: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//sb.gga.upf",
                Si: "https://raw.githubusercontent.com/Quantum-MultiScale/OFPP/refs/heads/main/EAC/upf/blps//si.gga.upf",
            },
            OEPP_recpot:{
                Ag : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Ag_lda.oe01.recpot",
                Al : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Al_lda.oe01.recpot",
                As : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/As_lda.oe04.recpot",
                Ba : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Ba_lda.oe01.recpot",
                Be : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Be_lda.oe02.recpot",
                Br : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Br_lda.oe01.recpot",
                Ca : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Ca_lda.oe01.recpot",
                Cd : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Cd_lda.oe01.recpot",
                Cs : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Cs_lda.oe02.recpot",
                Ga : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Ga_lda.oe04.recpot",
                Ge : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Ge_lda.oe02.recpot",
                Hg : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Hg_lda.oe01.recpot",
                I : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/I_lda.oe01.recpot",
                In : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/In_lda.oe03.recpot",
                K : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/K_lda.oe01.recpot",
                Li : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Li_lda.oe02.recpot",
                Mg : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Mg_lda.oe01.recpot",
                Na : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Na_lda.oe02.recpot",
                P : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/P_lda.oe01.recpot",
                Sb : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Sb_lda.oe03.recpot",
                Se : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Se_lda.oe01.recpot",
                Si : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Si_lda.oe01.recpot",
                S : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/S_lda.oe01.recpot",
                Sn : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Sn_lda.oe04.recpot",
                Sr : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Sr_lda.oe1.recpot",
                Te : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Te_lda.oe03.recpot",
                Zn : "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/RECPOT/Zn_lda.oe03.recpot",
            },
            OEPP_upf:{
                Ag : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Ag_OEPP_PZ.UPF",
                Al : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Al_OEPP_PZ.UPF",
                As : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//As_OEPP_PZ.UPF",
                Ba : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Ba_OEPP_PZ.UPF",
                Be : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Be_OEPP_PZ.UPF",
                Br : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Br_OEPP_PZ.UPF",
                Ca : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Ca_OEPP_PZ.UPF",
                Cd : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Cd_OEPP_PZ.UPF",
                Cs : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Cs_OEPP_PZ.UPF",
                Ga : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Ga_OEPP_PZ.UPF",
                Ge : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Ge_OEPP_PZ.UPF",
                Hg : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Hg_OEPP_PZ.UPF",
                I :  "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//I_OEPP_PZ.UPF",
                In : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//In_OEPP_PZ.UPF",
                K :  "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//K_OEPP_PZ.UPF",
                Li : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Li_OEPP_PZ.UPF",
                Mg : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Mg_OEPP_PZ.UPF",
                Na : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Na_OEPP_PZ.UPF",
                P :  "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//P_OEPP_PZ.UPF",
                Sb : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Sb_OEPP_PZ.UPF",
                Se : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Se_OEPP_PZ.UPF",
                Si : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Si_OEPP_PZ.UPF",
                S :  "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//S_OEPP_PZ.UPF",
                Sn : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Sn_OEPP_PZ.UPF",
                Sr : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Sr_OEPP_PZ.UPF",
                Te : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Te_OEPP_PZ.UPF",
                Zn : "https://gitlab.com/wenhui/OEPP/-/blob/master/OEPP/UPF//Zn_OEPP_PZ.UPF",
            },
            OEPP_cpi:{
                Ag: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi/Ag_local.cpi",
                Ba: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Ba_local.cpi",
                Be: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Be_local.cpi",
                Br: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Br_local.cpi",
                Ca: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Ca_local.cpi",
                Cd: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Cd_local.cpi",
                Cs: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Cs_local.cpi",
                Ge: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Ge_local.cpi",
                Hg: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Hg_local.cpi",
                I_: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//I_local.cpi",
                In: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//In_local.cpi",
                K_: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//K_local.cpi",
                Li: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Li_local.cpi",
                Na: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Na_local.cpi",
                P_: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//P_local.cpi",
                Sb: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Sb_local.cpi",
                Se: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Se_local.cpi",
                S_: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//S_local.cpi",
                Sn: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Sn_local.cpi",
                Sr: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Sr_local.cpi",
                Te: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Te_local.cpi",
                Zn: "https://gitlab.com/wenhui/OEPP/-/raw/master/OEPP/cpi//Zn_local.cpi",

            },
    };

    document.addEventListener("DOMContentLoaded", function () {
        const select = document.getElementById("pseudopotentials");

        function updateLinks(type) {
            const urls = links[type] || {};

            document.querySelectorAll(".element a").forEach(anchor => {
                const symbol = anchor.textContent.trim();
                const url = urls[symbol];
                if (typeof url === "string" && url.trim() !== "") {
                    anchor.href = url;
                    anchor.target = "_blank";
                    anchor.classList.remove("inactive");
                } else {
                    anchor.removeAttribute("href");
                    anchor.removeAttribute("target");
                    anchor.classList.add("inactive");
                }
            });
        }

        select.addEventListener("change", () => updateLinks(select.value));
        updateLinks(select.value); // initialize
    });
    </script>
