#!/usr/bin/env python
# encoding: utf-8
"""
Simple object to represent a given atom.  

File: simple_atom_object.py
Author: Amin.T

"""
#import sys,os
from reaction_prediction.atom.utils import *

from rpCHEM.Common.Util import splitCompositeSmilesToList, molBySmiles
from rpCHEM.Common.Util import joinSmilesListToCompositeSmiles
from openeye.oechem import OEAssignAromaticFlags, OEAroModelMMFF, OEClearAromaticFlags
from openeye.oechem import OEPerceiveChiral, OEPerceiveSymmetry
from openeye.oechem import OEFindRingAtomsAndBonds
#from rpCHEM.Common.Util import standardizeSmiles
from rpCHEM.Common.Util import createStdAtomMapSmiString, clearAtomMapsSmiStr
from rpCHEM.Common.CanonicalAtomMapSmiles import canonicalizeAtomMapSmiString, createCanonicalAtomMapSmiString
from rpCHEM.Common.MolExt import clearAtomMaps, removeNonsenseStereo
from rpCHEM.Common.MolExt import setSingleExplicitHydrogens, getAtmMappedSmilesFromCompositeSmiles


class SimpleAtomObject(object):
    """Simple object to capture a representation of an atom.
    
    Basically, given an atom object, store a copy of the atom object, full smiles, and the smiles of 
    only connected component containing self.
    """
    def __init__(self, oeAtom, oeMol, symmetricAtoms):
        """Set the basic params"""
        self.atom = oeAtom
        self.mol = oeMol
        
        self.symmetricAtoms = symmetricAtoms
        
        self.fullSmiles = None
        self.connectedSmiles = None
        self.connectedNonMappedSmiles = None
        
        clearAtomMaps(self.mol)
        self.atom.SetMapIdx(1)
        self.fullSmiles = createCanonicalAtomMapSmiString(self.mol)
        self.atom.SetMapIdx(0)
        self.connectedSmiles = getAtmMappedSmilesFromCompositeSmiles(self.fullSmiles, clearMaps=False)
        self.connectedSmiles = canonicalizeAtomMapSmiString(self.connectedSmiles)
        self.connectedNonMappedSmiles = clearAtomMapsSmiStr(self.connectedSmiles)
        self.connectedKekuleSmiles = canonicalKekule(self.connectedSmiles)
        
        # Eventual ML stuff
        self.rawFeatDict = None
        self.normFeatDict = None
        self.fill_pred_value = None
        self.unfill_pred_value = None
        self.fillPredDecision = None
        self.unfillPredDecision = None
    
    def __str__(self):
        """Just set this to return the connectedSmiles"""
        return self.connectedSmiles
    
    def __repr__(self):
        return "'%s'" % self.__str__();

    def neighbors(self, bond_order=None, include_h=False):
        """
        Return neighbor OEAtomBase objects.
        - bond_order: set to 1 for single bonds, 2 for double, etc.; None = any bond
        - include_h: include hydrogens if True
        """
        out = []
        for bond in self.atom.GetBonds():
            if bond_order is not None and bond.GetOrder() != bond_order:
                continue
            nbr = bond.GetNbr(self.atom)
            if not include_h and nbr.GetAtomicNum() == 1:
                continue
            out.append(nbr)
        return out

    # --- neighbors as SimpleAtomObject wrappers ---
    def neighbor_wrappers(self, bond_order=1, include_h=True):
        """
        Return neighbor atoms wrapped as SimpleAtomObject.
        """
        wrappers = []
        for nbr in self.neighbors(bond_order=bond_order, include_h=include_h):
            # If you want symmetry handled like in atomObjFromReactantSmi, you can
            # just seed with the single atom here; adjust if you need full symmetry groups.
            wrappers.append(SimpleAtomObject(nbr, self.mol, [nbr]))
        return wrappers
    
    @staticmethod
    def atomObjFromReactantSmi(reactantSmiles):
        """Convenience method to make a list of SimpleAtomObjects"""
        """Given a single reactant (connected component), return the list of atoms."""
        # First, ensure a single copy!
        #reactantSmiles = canonicalKekule(reactantSmiles)
        
        reactantSmiles = clearAtomMapsSmiStr(reactantSmiles)
        mol = molBySmiles(reactantSmiles)
        OEAssignAromaticFlags(mol, OEAroModelMMFF)
        reactantSmiles = createCanonicalAtomMapSmiString(mol)
        smilesSet = set(splitCompositeSmilesToList(reactantSmiles))
        newReactantSmiles = joinSmilesListToCompositeSmiles(list(smilesSet))
        newReactantSmiles = canonicalizeAtomMapSmiString(newReactantSmiles)
        
        mol = molBySmiles(newReactantSmiles);
        setSingleExplicitHydrogens(mol)
        OEAssignAromaticFlags(mol, OEAroModelMMFF)
        clearAtomMaps(mol)
        OEPerceiveChiral(mol)
        removeNonsenseStereo(mol)
        OEPerceiveSymmetry(mol)
        OEFindRingAtomsAndBonds(mol)
        
        resBySymmetryClass = {}
        seenSymClassSet = set([])
        for atm in mol.GetAtoms():
            theSymClass = atm.GetSymmetryClass()
            if theSymClass not in resBySymmetryClass:
                resBySymmetryClass[theSymClass] = SimpleAtomObject(atm, mol, [atm])
            else:
                resBySymmetryClass[theSymClass].symmetricAtoms.append(atm)
        return resBySymmetryClass.values()
    
    
    @staticmethod
    def combineAtomObjListToSmiles(atomObjList):
        """Given some 'filtered' atomObjs in a list, return a composite smiles with all of these atoms labeled."""
        if len(atomObjList) == 0:
            return None
        mol = atomObjList[0].mol
        clearAtomMaps(mol)
        for atomObj in atomObjList:
            [atm.SetMapIdx(1) for atm in atomObj.symmetricAtoms]
            #atomObj.atom.SetMapIdx(1)
        smi = createStdAtomMapSmiString(mol)
        clearAtomMaps(mol)
        return smi

    @staticmethod
    def combineAtomObjListToSmiles_diff_indices(atomObjList):
        """same as combineAtomObjListToSmiles but assign differenet numbers to each simpleAtomObj"""
        if len(atomObjList) == 0:
            return None
        mol = atomObjList[0].mol
        clearAtomMaps(mol)
        i = 1
        for atomObj in atomObjList:
            [atm.SetMapIdx(i) for atm in atomObj.symmetricAtoms]
            #atomObj.atom.SetMapIdx(1)
            i+=1
        smi = createStdAtomMapSmiString(mol)
        clearAtomMaps(mol)
        return smi
