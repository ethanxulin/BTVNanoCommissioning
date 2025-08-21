import awkward as ak
import numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights

from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
)

from BTVNanoCommissioning.helpers.func import update, dump_lumi, PFCand_link, add_discriminators
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.histogrammer import histogrammer, histo_writter
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_idiso,
    ele_mvatightid,
    softmu_mask,
    MET_filters,
)


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        selectionModifier="DYM",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        self.selMod = selectionModifier
        # Load corrections
        self.SF_map = load_SF(self._year, self._campaign)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        events = missing_branch(events)
        shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")

        isMu = False
        isEle = False

        if self.selMod == "ttdilep_sf_2Dcalib_2mu":
            triggers = ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8"]
            # dxySigcut = 1.0
            dxySigcut = 1.0
            muNeEmSum = 0.7
            isMu = True
        elif self.selMod == "ttdilep_sf_2Dcalib_2e":
            triggers = ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"]
            dxySigcut = 0.0
            muNeEmSum = 1.0
            isEle = True
        else:
            raise ValueError(self.selMod, "is not a valid selection modifier.")

        histname = {
            "ttdilep_sf_2Dcalib_2mu": "ttdilep_sf_2Dcalib_2mu",
            "ttdilep_sf_2Dcalib_2e": "ttdilep_sf_2Dcalib_2e"
        }
        output = {} if self.noHist else histogrammer(events, histname[self.selMod])


        ### This part is see is data or MC ###
        if isRealData:
            output["sumw"] = len(events)
        else:
            output["sumw"] = ak.sum(events.genWeight)

        ####################
        #    Selections    #
        ####################

        ### Lumimask ###
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        # HLT
        ### Basically trigger cuts here ###
        req_trig = HLT_helper(events, triggers)
        
        ##### This is new here????? #####
        ##### MET cut #####
        req_metfilter = MET_filters(events, self._campaign)
        req_metpt = events.PuppiMET.pt > 50

        # Muon cuts
        dilep_mu = events.Muon[(events.Muon.pt > 12) & mu_idiso(events, self._campaign)]

        ### Essentially, here is applying a booolean array to Muon at particle level ###
        ### Not event level ###

        muons = events.Muon[mu_idiso(events, self._campaign)]
        # dilep_mu = events.Muon[mu_idiso(events, self._campaign)]
        muons_sorted = muons[ak.argsort(muons.pt, ascending=False)]
        lead_mu = ak.firsts(muons_sorted)
        sublead_mu = ak.firsts(muons_sorted[:, 1:])
        lead_pt_mu = ak.fill_none(lead_mu.pt, 0.0)
        sublead_pt_mu = ak.fill_none(sublead_mu.pt, 0.0)
        lead_mask_mu = (lead_pt_mu > 20)
        sublead_mask_mu = (sublead_pt_mu > 12)
        mu_pt_mask = lead_mask_mu & sublead_mask_mu

        # events.Muon = events.Muon[
        #     (events.Muon.pt > 30) & mu_idiso(events, self._campaign)
        # ] earlier definition for comparision

        ### New Muon cuts ###
        # muons = events.Muon
        # muons = muons[ak.argsort(muons.pt, ascending=False)] ### Sorting Muons pt
        # dilep_mu = muons[(muons.pt > 12) & mu_idiso(events, self._campaign)] ### Get dilepton muon events that pass the cuts, pt > 12

        # has_two_mus = ak.num(muons.pt, axis=1) >= 2 ### Get mask for events with at least 2 muons
        
        # muons_padded = ak.pad_none(muons, 2)  # pad to 2 so both [0] and [1] exist as None if missing
        # lead_pt_mu = ak.fill_none(muons_padded[:, 0].pt, 0.0) ### Get leading muon pt, if no muon, then 0.0
        # sublead_pt_mu = ak.fill_none(muons_padded[:, 1].pt, 0.0) ### Get sub-leading muon pt, if no muon, then

        # # lead_pt_mu = ak.fill_none(muons[:, 0].pt, 0.0) ### Get leading muon pt, if no muon, then 0.0
        # # sublead_pt_mu = ak.fill_none(muons[:, 1].pt, 0.0) ### Get sub-leading muon pt, if no muon, then 0.0
        # # lead_pt_mu = ak.where(has_two_mus, muons[:, 0].pt, 0.0)
        # # sublead_pt_mu = ak.where(has_two_mus, muons[:, 1].pt, 0.0)
        
        # pt_mask_mu = (lead_pt_mu > 20) & (sublead_pt_mu > 12) ### Get mask for leading and sub-leading muon pt
        # mu_mask = has_two_mus & pt_mask_mu ### Combine the two masks to get events with at least 2 muons, and leading muon pt > 20, sub-leading muon pt > 12
        # dilep_mu = muons[mu_mask] ### Get dilepton muon events that pass the cuts






        # Electron cuts

        # dilep_mu = events.Muon[(events.Muon.pt > 12) & mu_idiso(events, self._campaign)]

        ### Essentially, here is applying a booolean array to Muon at particle level ###
        ### Not event level ###

        dilep_ele = events.Electron[
            (events.Electron.pt > 15) & ele_mvatightid(events, self._campaign)
        ]
        
        electrons = events.Electron[ele_mvatightid(events, self._campaign)]
        # dilep_mu = events.Muon[mu_idiso(events, self._campaign)]
        electrons_sorted = electrons[ak.argsort(electrons.pt, ascending=False)]
        lead_el = ak.firsts(electrons_sorted)
        sublead_el = ak.firsts(electrons_sorted[:, 1:])
        lead_pt_el = ak.fill_none(lead_el.pt, 0.0)
        sublead_pt_el = ak.fill_none(sublead_el.pt, 0.0)
        lead_mask_el = (lead_pt_el > 20)
        sublead_mask_el = (sublead_pt_el > 12)
        el_pt_mask = lead_mask_el & sublead_mask_el

        # events.Electron = events.Electron[
        #     (events.Electron.pt > 30) & ele_cuttightid(events, self._campaign)
        # ]

        ### New Electron cuts ###
        # electrons = events.Electron
        # electrons = electrons[ak.argsort(electrons.pt, ascending=False)] ### Sorting Electrons pt
        # dilep_ele = electrons[(electrons.pt > 15) & ele_mvatightid(events, self._campaign)] ### Get dilepton electron events that pass the cuts, pt > 15
        # has_two_eles = ak.num(electrons.pt, axis=1) >= 2 ### Get mask for events with at least 2 electrons
       
        # lead_pt_ele = ak.where(has_two_eles, electrons[:, 0].pt, 0.0) ### Get leading electron pt, if no electron, then 0.0
        # sunlead_pt_ele = ak.where(has_two_eles, electrons[:, 1].pt, 0.0) ### Get sub-leading electron pt
        
        # electrons_padded = ak.pad_none(electrons, 2)  # pad to 2 so both [0] and [1] exist as None if missing
        # lead_pt_ele = ak.fill_none(electrons_padded[:, 0].pt, 0.0) ### Get leading electron pt, if no electron, then 0.0
        # sublead_pt_ele = ak.fill_none(electrons_padded[:, 1].pt, 0.0) ### Get sub-leading electron pt, if no electron, then 0.0


        # pt_mask_ele = (lead_pt_ele > 27) & (sublead_pt_ele > 15) ### Get mask for leading and sub-leading electron pt
        # ele_mask = has_two_eles & pt_mask_ele ### Combine the two masks to get events with at least 2 electrons, and leading electron pt > 20, sub-leading electron pt
        # dilep_ele = electrons[ele_mask] ### Get dilepton electron events that pass the cuts


        


        if isMu: # depending on which workflow you are applying
            muons_selected = ak.where(mu_pt_mask, muons_sorted, ak.Array([[]]*len(muons_sorted)))
            thisdilep = muons_selected
            # thisdilep = dilep_mu
            otherdilep = dilep_ele
            req_pt = mu_pt_mask
        else:
            electrons_selected = ak.where(el_pt_mask, electrons_sorted, ak.Array([[]]*len(electrons_sorted)))
            thisdilep = electrons_selected
            otherdilep = dilep_mu
            req_pt = el_pt_mask

        ### dilepton cut ###
        pos_dilep = thisdilep[thisdilep.charge > 0] # Picks out the positively charged leptons in thisdilep array, could be more than 1
        neg_dilep = thisdilep[thisdilep.charge < 0] # Picks out the negatively charged leptons in thisdilep array, could be more than 1
        req_pl = ak.count(pos_dilep.pt, axis=1) >= 1 # picks out the positively charged leptons events that contains atleast one +ve charged lepton
        req_nl = ak.count(neg_dilep.pt, axis=1) >= 1 # picks out the -vely charged leptons events that contains atleast one -ve charged lepton
        req_dilep_chrg = ak.num(thisdilep.charge) >= 2 # picks out all leprons that pass the kinematic cuts, and has number of DESIRED charged leptons at least 2
        req_otherdilep_chrg = ak.num(otherdilep.charge) == 0 # the number of other charged lepton must be zero (like desired muon, then wish electron to be 0)
        req_dilep = ak.fill_none(
            req_pl & req_nl & req_dilep_chrg & req_otherdilep_chrg,
            False,
            axis=-1,
        )
        ### basically the leptons cut, which is the combination of:
        ### at least one +ve charged desired lepton
        ### at least one -ve charged desired lepton
        ### at least two desired leptons
        ### zero lepton that is not desired



        ###
        ### The pt of lepton here used is the hard muon we asked pt to be greater than 12
        ### and electron pt to be greater than 15 earlier
        ### By which to be isolated from jets, and jets to be isolated from these hard leptons
        ### This part has nothing to do with the soft muon later we need to looking for later

        pl_iso = ak.all(
            events.Jet.metric_table(pos_dilep) > 0.4, axis=2, mask_identity=True
        )
        nl_iso = ak.all(
            events.Jet.metric_table(neg_dilep) > 0.4, axis=2, mask_identity=True
        )
        jet_sel = ak.fill_none(
            jet_id(events, self._campaign) & pl_iso & nl_iso,
            False,
            axis=-1,
        )

        pos_dilep = ak.pad_none(pos_dilep, 1, axis=1)
        ### input are just number 1? or True?
        neg_dilep = ak.pad_none(neg_dilep, 1, axis=1)


        ### Basically dilepmass cuts here ###
        ### Picks out the first lepton inside the +ve charged and -ve charged, and form their invariant mass, could play around with it ###
        dilep_mass = pos_dilep[:, 0] + neg_dilep[:, 0]
        req_dilepmass = (
            # (dilep_mass.mass > 81) & (dilep_mass.mass < 101) & (dilep_mass.pt > 15) # DY cut
            ((dilep_mass.mass < 81) | (dilep_mass.mass > 101)) & (dilep_mass.pt > 15)
            # (dilep_mass.mass > 101) & (dilep_mass.pt > 15)
            ### No need lorentz vector??? ###
        )

        ### Need to understand this part ###


        ### Jet cuts ###
        pl_iso = ak.all(
            events.Jet.metric_table(pos_dilep[:, 0]) > 0.4, axis=2, mask_identity=True
        )
        nl_iso = ak.all(
            events.Jet.metric_table(neg_dilep[:, 0]) > 0.4, axis=2, mask_identity=True
        )
        event_jet = events.Jet[
            ak.fill_none(
                jet_id(events, self._campaign) & pl_iso & nl_iso,
                False,
                axis=-1,
            )
        ]
        req_jets = ak.count(event_jet.pt, axis=1) >= 2
        ### At least two jets ###


        # jetsel = ak.fill_none(
        #     jet_id(events, self._campaign)
        #     & (
        #         ak.all(
        #             events.Jet.metric_table(events.Muon) > 0.4,
        #             axis=2,
        #             mask_identity=True,
        #         )
        #     )
        #     & (
        #         ak.all(
        #             events.Jet.metric_table(events.Electron) > 0.4,
        #             axis=2,
        #             mask_identity=True,
        #         )
        #     ),
        #     False,
        # )
        # event_jet = events.Jet[jetsel]
        # req_jets = ak.num(event_jet.pt) >= 2


        
        ## Soft Muon cuts
        soft_muon = events.Muon[
            softmu_mask(events, self._campaign)
            & (abs(events.Muon.dxy / events.Muon.dxyErr) > dxySigcut)
        ]
        req_softmu = ak.count(soft_muon.pt, axis=1) >= 1
        #print(req_softmu)
        mujetsel = ak.fill_none(
            (
                (ak.all(event_jet.metric_table(soft_muon) <= 0.4, axis=2))
                & ((event_jet.muonIdx1 != -1) | (event_jet.muonIdx2 != -1))
                & ((event_jet.muEF + event_jet.neEmEF) < muNeEmSum)
                & (event_jet.pt > 20)
                & ((event_jet.pt / event_jet.E) > 0.03)
            ),
            False,
            axis=-1,
        )
        mujetsel2 = ak.fill_none(
            (
                ((events.Jet.muEF + events.Jet.neEmEF) < muNeEmSum)
                & (
                    ak.all(
                        events.Jet.metric_table(soft_muon) <= 0.4,
                        axis=2,
                        mask_identity=True,
                    )
                )
                & ((events.Jet.muonIdx1 != -1) | (events.Jet.muonIdx2 != -1))
            ),
            False,
            axis=-1,
        )
        soft_muon = ak.pad_none(soft_muon, 1, axis=1)
        soft_muon["dxySig"] = soft_muon.dxy / soft_muon.dxyErr

        ## Muon-jet cuts
        event_jet["isMuonJet"] = mujetsel
        print("event_jet", event_jet, len(event_jet))
        mu_jet = event_jet[mujetsel]
        print("mu_jet", mu_jet, len(mu_jet))
        print(ak.sum(ak.num(mu_jet.pt, axis=1)>0))
        otherjets = event_jet[~mujetsel]
        req_mujet = ak.num(mu_jet.pt, axis=1) >= 1
        mu_jet = ak.pad_none(mu_jet, 1, axis=1)













        # event_jet = ak.pad_none(event_jet, 1, axis=1)

        # store jet index for PFCands, create mask on the jet index
        jetindx = ak.mask(
            ak.local_index(events.Jet.pt),
            jet_sel == 1,
        )
        jetindx = ak.pad_none(jetindx, 1)
        jetindx = jetindx[:, 0]




        ### All cuts ###
        # event_level = ak.fill_none(
        #     req_lumi & req_trig & req_dilep & req_dilepmass & req_jets & req_metfilter,
        #     False,
        # )
        # event_level = ak.fill_none(
        #     req_lumi & req_trig & req_dilep & req_dilepmass & req_jets & req_metfilter & req_softmu & req_mujet & req_metpt,
        #     False,
        # )
        
        event_level = ak.fill_none(
            req_lumi & req_trig & req_dilep & req_dilepmass & req_jets & req_softmu & req_mujet & req_pt & req_metfilter & req_metpt,
            False,
        )
        print("event_level", ak.sum(event_level))
        # event_level = ak.fill_none(
        #     req_lumi & req_trig & req_dilep & req_dilepmass & req_jets & req_pt & req_metfilter & req_metpt,
        #     False,
        # )
        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}
        ####################
        # Selected objects #
        ####################
        sposmu = pos_dilep[event_level][:, 0]
        snegmu = neg_dilep[event_level][:, 0]
        sz = sposmu + snegmu


        smuon_jet = mu_jet[event_level]
        print("smuon_jet", smuon_jet)
        #nmujet = ak.count(smuon_jet.pt, axis=1)


        sel_jet = event_jet[event_level][:, 0]

        sel_mu = ak.concatenate([sposmu, snegmu])
        smu = ak.zip(
            {
                b: ak.Array(np.reshape(sel_mu[b].to_numpy(), (len(sposmu[b]), 2)))
                for b in sposmu.fields
            }
        )
        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = event_jet[event_level]
        ssmu = soft_muon[event_level][:, 0]

        ### if higher purity for jets that contain soft muon
        ### maybe should use mu_jet here, not event_jet?





        if isMu:
            pruned_ev["MuonPlus"] = sposmu
            pruned_ev["MuonMinus"] = snegmu
            pruned_ev["posl"] = sposmu
            pruned_ev["negl"] = snegmu
            pruned_ev["SelMuon"] = smu
            # kinOnly = ["SelMuon", "MuonPlus", "MuonMinus"]
        else:
            pruned_ev["ElectronPlus"] = sposmu
            pruned_ev["ElectronMinus"] = snegmu
            pruned_ev["posl"] = sposmu
            pruned_ev["negl"] = snegmu
            pruned_ev["SelElectron"] = smu
            # kinOnly = ["SelElectron", "ElectronPlus", "ElectronMinus"]
        pruned_ev["SoftMuon"] = ssmu
        pruned_ev["dilep"] = sz
        pruned_ev["dilep", "pt"] = pruned_ev.dilep.pt
        pruned_ev["dilep", "eta"] = pruned_ev.dilep.eta
        pruned_ev["dilep", "phi"] = pruned_ev.dilep.phi
        pruned_ev["dilep", "mass"] = pruned_ev.dilep.mass
        pruned_ev["njet"] = ak.count(event_jet[event_level].pt, axis=1)
        print("pruned_ev[njet]", pruned_ev["njet"])
        print("line holder")
        #pruned_ev["nmujet"] = ak.count(mu_jet[event_level].pt, axis=1)
        #pruned_ev["MuonJet"] = smuon_jet
        print("smuon_jet", smuon_jet)
        print("line holder")
        # print("number_smuon_jet", ak.sum(smuon_jet.pt>0))
        print("number_smuon_jet", ak.count(smuon_jet.pt, axis=1))
        print("space holder")
        # pruned_ev["nmujet"] = ak.count(smuon_jet.pt, axis=1)
        # pruned_ev["nmujet"] = ak.sum(smuon_jet.pt>0)
        pruned_ev["nmujet"] = ak.count(smuon_jet.pt, axis=1)
        smuon_jet = smuon_jet[:, 0] ### Has to define after nmujet!!! otherwise axis=1 won't work for acount in nmujet
        # smuon_jet = smuon_jet[:, 0]
        pruned_ev["MuonJet"] = smuon_jet
        print("pruned_ev[nmujet]", pruned_ev["nmujet"])
        print("Don't block the way")


        # pruned_ev["soft_l_ptratio"] = ssmu.pt / smuon_jet.pt


##### Question, how it define neg and pos ones #####
        pruned_ev["dr_mu1jet"] = sposmu.delta_r(sel_jet)
        pruned_ev["dr_mu2jet"] = snegmu.delta_r(sel_jet)
        # Find the PFCands associate with selected jets. Search from jetindex->JetPFCands->PFCand
        if "PFCands" in events.fields:
            pruned_ev["PFCands"] = PFCand_link(events, event_level, jetindx)

        if "2Dcalib" in self.selMod:
            taggers =  ["DeepFlav", "PNet", "UParTAK4"] #RobustParTAK4
            for tagger in taggers:
                pruned_ev["SelJet"] = add_discriminators(pruned_ev["SelJet"], tagger)
                pruned_ev["MuonJet"] = add_discriminators(pruned_ev["MuonJet"], tagger)
       
        ####################
        #     Output       #
        ####################
        # Configure SFs
        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Configure histograms
        if not self.noHist:
            output = histo_writter(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )
        # Output arrays
        if self.isArray:
            array_writer(
                self, pruned_ev, events, weights, systematics, dataset, isRealData
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
