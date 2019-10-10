//
//  withpetsc.cpp
//
//
//  Created by Purnima Ghale on 4/4/18.
// FINAL version for BW-Exploratory-Allocation
// Computes density matrix using KPE+SP2
//

//# include "withpetsc.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <petscksp.h>
#include <petsctime.h>

using namespace std;
static char help[] = "Solves for density matrix .\n\n";

PetscInt maxN_A = 20;
PetscInt maxsp2;
const PetscScalar PI=3.141592653589793238463;
PetscInt nSize, Nelectrons;
PetscInt L_lattice, M_lattice, N_lattice, Natoms;
// for local variables:
PetscInt localHSize;            // local number of rows of hamiltonian
PetscScalar* localXs;
PetscScalar* localYs;
PetscScalar* localZs;
PetscInt* localAtoms;
PetscInt* localBeginIndex;       // begin index of this local atom on the global Hamiltonian
PetscInt* localEndIndex;
PetscInt ni;                     // Number of atoms owned by local processor
PetscInt       rbegin, rend;
PetscScalar  muval;
double* alphavals;
PetscInt nRandoms;
PetscInt       MPI_RANK;
// for neighbors:
PetscInt* nN;       // number of neighbors of each local atom
PetscScalar* xsN;   // xs of neighbors of atoms, size = [maxN_A*numberOfLocalAtoms]
PetscScalar* ysN;
PetscScalar* zsN;
PetscInt* aTN;
PetscInt* bIN;
PetscInt* eIN;
PetscInt* gAIN;

// for petsc variables that are global:
Mat H, Htotal, deltaH;
Vec Xs, Ys, Zs, netCharges, netQNeighbors;
Vec LambdaMaxs, LambdaMins;
Vec diagonalH0G, offDiagonalH0G;            // diagonal, off-diagonal parts for gershgorin
Vec diagonaldHG, offDigonaldHG, diagP;      // diagonal, off-diagonal parts of deltaH, diagP of density matrix:
PetscErrorCode ierr;

// PetscInterpolation variables:
//
//  get4by4matrix.cpp
//
//
//  Created by Purnima Ghale on 4/11/18.
//
//
PetscScalar Si_Si_e[20][4] = {
    {17.865,      -5.7398 ,      17.262,      -13.794},
    {87.114,       5.8664   ,    39.685    ,  -7.1328},
    {-722.33,      -52.111  ,    -283.25   ,    27.816},
    {66.059  ,    -3.2651   ,    34.498    ,  -27.631},
    {-613.94  ,    -6.6631  ,    -41.298   ,    12.075},
    {-727.27 ,      1.3757  ,    -112.62   ,    5.9498},
    {246.1   ,   -84.273    ,   241.71     , -164.58},
    {-1564.3 ,      4.1541  ,    -179.83   ,    22.449},
    {-589.53 ,     -7.9678  ,      215.3   ,   -36.682},
    {-1020.8 ,       33.73  ,     7.8809   ,   -87.911},
    {-812.85 ,     -33.408  ,     13.727   ,    158.22},
    {1189.7  ,     49.751   ,    54.842    ,  -56.155},
    {-96.931 ,      8.9531  ,     102.98   ,   -91.347},
    {860.35  ,     1.2204   ,   -44.669    ,   12.487},
    {3551.6  ,     9.5425   ,   -32.771    ,   10.438},
    {468.97  ,     1.6056   ,   -128.73    ,  -1.5888},
    {-317.44 ,      6.9669  ,     133.21   ,    82.532},
    {517.74  ,     47.052   ,   -56.922    ,   1.1376},
    {210.8   ,   -18.241    ,  -18.441     ,  80.163},
    {-443.57  ,     36.135   ,    52.835  ,    -13.413}
};

PetscScalar O_O_e[20][4] = {
    {5.5526  ,   -0.82118  ,      3.813 ,     -1.2323},
    {-31.886 ,      10.518  ,    -41.071   ,    52.454},
    {233.96  ,    -39.064   ,     272.9   ,   -407.19},
    {-60.015 ,      61.253  ,    -128.95  ,    -294.22},
    {213.73  ,    -204.34   ,    410.72   ,    1263.6},
    {49.054  ,     32.239   ,   -78.906   ,     -1060},
    {-73.081  ,     132.93  ,    -81.269  ,     1956.2},
    {33.537   ,   -13.278   ,   -137.34   ,   -384.47},
    {78.858   ,    165.14   ,   -273.61   ,   -873.53},
    {-167.41  ,     68.088  ,    -277.49  ,     642.65},
    {75.731   ,    7.9468   ,   -767.04   ,   -615.24},
    {-63.795  ,    -44.453   ,   -288.57  ,     988.73},
    {-78.765  ,     32.997   ,   -76.092  ,    -811.61},
    {124.63   ,    -144.7    ,   -189.3   ,    554.51},
    {40.615   ,   -149.44    ,   780.52   ,    379.23},
    {82.116   ,   -29.881    ,    410.4   ,   -93.407},
    {-80.656  ,    -54.736   ,     203.1  ,    -131.08},
    {-74.163  ,     16.814   ,    710.57  ,    -213.88},
    {7.3441   ,     20.02    ,  -100.83  ,     128.52},
    {-93.219  ,     6.9584   ,    267.84 ,     -242.74}
};

PetscScalar Si_O_e[20][4]= {
    {13.778  ,    -2.7061  ,     8.6477 ,     -7.6931},
    {9.0798   ,    19.494     ,  -22.43    ,   52.885},
    {-15.173  ,    -145.38    ,   262.06   ,   -468.23},
    {9.4007   ,   -23.026    ,   223.86    ,  -100.16},
    {40.783   ,    126.17    ,  -785.64    ,   228.63},
    {23.249   ,   -34.501    ,   334.88    ,  -253.56},
    {-17.829  ,    -75.476   ,    156.18   ,   -638.68},
    {18.37    ,  -22.003     ,  10.705     , -312.46},
    {1.7289   ,    75.972    ,   572.55    ,  -1010.6},
    {-30.424  ,     17.518   ,   -55.884   ,   -478.46},
    {-31.663  ,    -36.928   ,   -349.49   ,   -688.24},
    {-82.425  ,    -26.609   ,   -390.25   ,   -296.52},
    {35.07    ,   77.948    ,  -143.41     ,  157.81},
    {-27.066  ,    -22.899  ,    -390.79   ,    580.81},
    {-20.555  ,    -140.91   ,   -152.48   ,    401.59},
    {49.585   ,   -5.9967   ,    502.82    ,   768.76},
    {-8.7806  ,     109.97  ,    -113.52   ,    980.42},
    {37.13    ,  -18.019    ,  -148.42     ,  243.34},
    {-20.674  ,     15.652  ,     904.76   ,    317.15},
    {16.513   ,    39.501   ,    -630.1    ,    84.78}
};


void get4by4HmatrixElements(PetscInt atom1, PetscInt atom2, PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar* hvalues){
    // elements of tb-model obtained from dftb, other models possible
    // extrapolated to functional form:
    PetscScalar rSq = (pow(x,2) + pow(y,2) + pow(z,2));
    PetscScalar r = sqrt(rSq);
    PetscScalar H[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    //cout<<"x : "<<x<<" y : "<<y<<" z : "<<z<<endl;
    if ((atom1==14 && atom2==8) ||(atom1==8 && atom2==14)){
        if (r<0.01){
            cout<<"atoms of 2 different types, closer than 0.01"<<endl;
            throw exception();
        }else{
            PetscScalar cp2, cp3, cp5, cp6;
            for (PetscInt p=1; p<=20; p++){  // parameters for analytical expression were constructed with p={1,2,3 ...} etc in matlab
                cp2 = Si_O_e[p-1][0];
                cp3 = Si_O_e[p-1][1];
                cp5 = Si_O_e[p-1][2];  //cout<<"cp5: "<<cp5<<endl;
                cp6 = Si_O_e[p-1][3];  //cout<<"cp6: "<<cp6<<endl;
                //cout<<" cp2 : "<<cp2<<" cp3 : "<<cp3<<" cp5 : "<<cp5<<" cp6 : "<<cp6<<endl;
                //cout<<H[0][0]<<"   "<<H[0][1]<<"   "<<H[0][2]<<"   "<<H[0][3]<<endl;
                
                H[0][0]= H[0][0] + cp6*cos(p*acos((r)/10))*exp(-p*(r));
                H[0][1]= H[0][1] + (cp5*x*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[0][2]= H[0][2] + (cp5*y*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[0][3]= H[0][3] + (cp5*z*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                
                
                H[1][0]= H[1][0] + -(cp5*x*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[1][1]= H[1][1] + (cp2*pow(x,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(x,2)/(rSq) - 1);
                H[1][2]= H[1][2] + (cp2*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[1][3]= H[1][3] + (cp2*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                
                
                H[2][0]= H[2][0] + -(cp5*y*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[2][1]= H[2][1] + (cp2*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[2][2]= H[2][2] + (cp2*pow(y,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(y,2)/(rSq) - 1);
                H[2][3]= H[2][3] + (cp2*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                
                
                H[3][0]= H[3][0] + -(cp5*z*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[3][1]= H[3][1] + (cp2*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[3][2]= H[3][2] + (cp2*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[3][3]= H[3][3] + (cp2*pow(z,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(z,2)/(rSq) - 1);
            }
        }
    }else if (atom1==14 && atom2==14){
        if (r<0.01){
            PetscScalar Ep = -0.15031380;
            PetscScalar Es = -0.39572506;
            PetscScalar SPE = -0.028927;
            PetscScalar Ud = 0.247609;
            
            H[0][0]=Es; H[1][1]=Ep; H[2][2]=Ep; H[3][3]=Ep;
        }else{
            PetscScalar cp2, cp3, cp5, cp6;
            for (PetscInt p=1; p<=20; p++){
                cp2 = Si_Si_e[p-1][0];  // cout<<"cp2: "<<cp2<<endl;
                cp3 = Si_Si_e[p-1][1];  // cout<<"cp3: "<<cp3<<endl;
                cp5 = Si_Si_e[p-1][2];  // cout<<"cp5: "<<cp5<<endl;
                cp6 = Si_Si_e[p-1][3];  // cout<<"cp6: "<<cp6<<endl;
                //cout<<" cp2 : "<<cp2<<" cp3 : "<<cp3<<" cp5 : "<<cp5<<" cp6 : "<<cp6<<endl;
                
                H[0][0]= H[0][0] + cp6*cos(p*acos((r)/10))*exp(-p*(r));
                H[0][1]= H[0][1] + (cp5*x*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[0][2]= H[0][2] + (cp5*y*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[0][3]= H[0][3] + (cp5*z*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                
                
                H[1][0]= H[1][0] + -(cp5*x*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[1][1]= H[1][1] + (cp2*pow(x,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(x,2)/(rSq) - 1);
                H[1][2]= H[1][2] + (cp2*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[1][3]= H[1][3] + (cp2*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                
                
                H[2][0]= H[2][0] + -(cp5*y*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[2][1]= H[2][1] + (cp2*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[2][2]= H[2][2] + (cp2*pow(y,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(y,2)/(rSq) - 1);
                H[2][3]= H[2][3] + (cp2*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                
                
                H[3][0]= H[3][0] + -(cp5*z*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[3][1]= H[3][1] + (cp2*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[3][2]= H[3][2] + (cp2*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[3][3]= H[3][3] + (cp2*pow(z,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(z,2)/(rSq) - 1);
            }
        }
    }else if (atom1==8 && atom2==8){
        if (r<0.01){
            PetscScalar Ep = -0.33213167;
            PetscScalar Es = -0.87883246;
            PetscScalar SPE = -0.05573427;
            PetscScalar Ud = 0.0;
            PetscScalar Up = 0.4954;
            H[0][0]=Es; H[1][1]=Ep; H[2][2]=Ep; H[3][3]=Ep;
        }else{
            PetscScalar cp2, cp3, cp5, cp6;
            
            for (PetscInt p=1; p<=20; p++){
                cp2 = O_O_e[p-1][0];   //cout<<"cp2: "<<cp2<<endl;
                cp3 = O_O_e[p-1][1];  // cout<<"cp3: "<<cp3<<endl;
                cp5 = O_O_e[p-1][2];  // cout<<"cp5: "<<cp5<<endl;
                cp6 = O_O_e[p-1][3];  // cout<<"cp6: "<<cp6<<endl;
                //cout<<" cp2 : "<<cp2<<" cp3 : "<<cp3<<" cp5 : "<<cp5<<" cp6 : "<<cp6<<endl;
                H[0][0]= H[0][0] + cp6*cos(p*acos((r)/10))*exp(-p*(r));
                H[0][1]= H[0][1] + (cp5*x*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[0][2]= H[0][2] + (cp5*y*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[0][3]= H[0][3] + (cp5*z*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                
                
                H[1][0]= H[1][0] + -(cp5*x*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[1][1]= H[1][1] + (cp2*pow(x,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(x,2)/(rSq) - 1);
                H[1][2]= H[1][2] + (cp2*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[1][3]= H[1][3] + (cp2*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                
                
                H[2][0]= H[2][0] + -(cp5*y*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[2][1]= H[2][1] + (cp2*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*y*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[2][2]= H[2][2] + (cp2*pow(y,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(y,2)/(rSq) - 1);
                H[2][3]= H[2][3] + (cp2*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                
                
                H[3][0]= H[3][0] + -(cp5*z*cos(p*acos((r)/10))*exp(-p*(r)))/(r);
                H[3][1]= H[3][1] + (cp2*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*x*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[3][2]= H[3][2] + (cp2*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - (cp3*y*z*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq);
                H[3][3]= H[3][3] + (cp2*pow(z,2)*cos(p*acos((r)/10))*exp(-p*(r)))/(rSq) - cp3*cos(p*acos((r)/10))*exp(-p*(r))*(pow(z,2)/(rSq) - 1);
            }
        }
    }else{
        cout<<"atom case not found, need to update source code to include case---- "<<endl;
        throw exception();
    }
    
    for (PetscInt i=0;i<4;i++){
        for (PetscInt j=0;j<4;j++){
            hvalues[i*4+j] = H[i][j];
        }
    }
    
}

PetscScalar cot(PetscScalar x){
    return (cos(x)/sin(x));
}

void ModifyNList(int p,double xi, double yi, double zi, int atomType, int beginIndex, int endIndex, int atomNumber){
    int whichNeighbor = nN[p];
    if (whichNeighbor>=maxN_A){
        cout<<"increase the max. allowed number of neighbors :"<<whichNeighbor<<"/"<<maxN_A<<"  "<< endl;
        throw exception();
    }else{
        xsN[p*maxN_A + whichNeighbor] = xi;
        ysN[p*maxN_A + whichNeighbor] = yi;
        zsN[p*maxN_A + whichNeighbor] = zi;
        aTN[p*maxN_A + whichNeighbor] = atomType;
        bIN[p*maxN_A + whichNeighbor] = beginIndex;
        eIN[p*maxN_A + whichNeighbor] = endIndex;
        gAIN[p*maxN_A + whichNeighbor] = atomNumber;
        nN[p] = nN[p] + 1;
    }
}

void getHaloAtoms(PetscInt nlocal, PetscInt minIndex, PetscInt maxIndex, const PetscScalar* localXs, const PetscScalar* localYs, const PetscScalar* localZs){
    //! get halo atoms or possible neighboring atoms in other processors while the Xs, Ys, Zs vectors are assembling.
    //! nlocal = number of local poPetscInts
    //! minIndex, maxIndex : indexes of atoms belonging to this process, that should not be considered halos
    //! localXs, localYs, localZs : local coordinates whose neighbors we should find.
    //! updates and changes std::vectors such as -- haloXs, haloYs, haloZs, haloAtomTypes, haloIndices, halobeginIndex, haloEndIndex
    PetscInt atomNumber=0;
    PetscScalar x2, y2, z2, x0, y0, z0, xi, yi, zi, AngToBohr=1.88973;
    PetscScalar a =9.725, b=a, c=10.7; // in Bohrs
    PetscScalar R1[3], R2[3], R3[3];
    PetscScalar periodicR1[3], periodicR2[3], periodicR3[3];
    
    R1[0] = a;                  R1[1] = 0.0;            R1[2] = 0.0;
    R2[0] = b*(-sin(PI/6));     R2[1]= b*cos(PI/6);       R2[2] = 0.0;
    R3[0] = 0.0;                R3[1] = 0.0;            R3[2] = c;
    
    periodicR1[0] = L_lattice*R1[0];   periodicR2[0] = M_lattice*R2[0];   periodicR3[0] = N_lattice*R3[0];
    periodicR1[1] = L_lattice*R1[1];   periodicR2[1] = M_lattice*R2[1];   periodicR3[1] = N_lattice*R3[1];
    periodicR1[2] = L_lattice*R1[2];   periodicR2[2] = M_lattice*R2[2];   periodicR3[2] = N_lattice*R3[2];
    
    PetscScalar distance, rcut=16.0;
    PetscInt beginIndex=0; PetscInt endIndex=0;
    PetscScalar xperiodic = 0.0, yperiodic = 0.0, zperiodic=0.0;
    //cout<<"periodicR1 : "<<periodicR1[0]<<"  "<<periodicR1[1]<<"   "<<periodicR1[2]<<endl;
    /*cout<<"periodicR2 : "<<periodicR2[0]<<"  "<<periodicR2[1]<<"   "<<periodicR2[2]<<endl;
    cout<<"periodicR3 : "<<periodicR3[0]<<"  "<<periodicR3[1]<<"   "<<periodicR3[2]<<endl;
    cout<<"the number of local atom is : "<<nlocal<<endl;
    */
    for (int lp=-1; lp<2; lp++){
        for (int mp=-1; mp<2; mp++){
	    int np=0;
           // for (int np=-1; np<2; np++){
                xperiodic = lp*periodicR1[0] + mp*periodicR2[0] + np*periodicR3[0];
                yperiodic = lp*periodicR1[1] + mp*periodicR2[1] + np*periodicR3[1];
                zperiodic = lp*periodicR1[2] + mp*periodicR2[2] + np*periodicR3[2];
      		//cout<<"xperiodic "<<xperiodic<<"  yperiodic "<<yperiodic<<"  zperiodic  "<<zperiodic<<endl;          
                for (PetscInt p=0; p<nlocal; p++){
                    x2 = localXs[p];    y2 = localYs[p];    z2 = localZs[p];
                    atomNumber = 0; beginIndex = 0; endIndex = 0; nSize = 0;
                    for (PetscInt i=0;i<L_lattice;i++){ for (PetscInt j=0;j<M_lattice; j++){ for (PetscInt k=0;k<N_lattice;k++){
                        x0 = i*R1[0] + j*R2[0] + k*R3[0] + xperiodic;
                        y0 = i*R1[1] + j*R2[1] + k*R3[1] + yperiodic;
                        z0 = i*R1[2] + j*R2[2] + k*R3[2] + zperiodic;
                        
                        // 1 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.4697*R1[0] + 0.0000*R2[0] + 0.0000*R3[0] + x0;
                        yi = 0.4697*R1[1] + 0.0000*R2[1] + 0.0000*R3[1] + y0;
                        zi = 0.4697*R1[2] + 0.0000*R2[2] + 0.0000*R3[2] + z0;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 14, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        
                        // 2 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.0000*R1[0] + 0.4697*R2[0] + 0.6777*R3[0] + x0;
                        yi = 0.0000*R1[1] + 0.4697*R2[1] + 0.6777*R3[1] + y0;
                        zi = 0.0000*R1[2] + 0.4697*R2[2] + 0.6777*R3[2] + z0;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 14, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        
                        // 3 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.5303*R1[0] + 0.5303*R2[0] + 0.3333*R3[0] + x0;
                        yi = 0.5303*R1[1] + 0.5303*R2[1] + 0.3333*R3[1] + y0;
                        zi = 0.5303*R1[2] + 0.5303*R2[2] + 0.3333*R3[2] + z0;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 14, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        
                        // 4 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.4133*R1[0] + 0.2672*R2[0] + 0.1188*R3[0] + x0;
                        yi = 0.4133*R1[1] + 0.2672*R2[1] + 0.1188*R3[1] + y0;
                        zi = 0.4133*R1[2] + 0.2672*R2[2] + 0.1188*R3[2] + z0;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 8, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        
                        // 5 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.2672*R1[0] + 0.4133*R2[0] + 0.5479*R3[0] + x0;
                        yi = 0.2672*R1[1] + 0.4133*R2[1] + 0.5479*R3[1] + y0;
                        zi = 0.2672*R1[2] + 0.4133*R2[2] + 0.5479*R3[2] + z0;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 8, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        
                        // 6 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.7328*R1[0] + 0.1461*R2[0] + 0.7855*R3[0] + x0;
                        yi = 0.7328*R1[1] + 0.1461*R2[1] + 0.7855*R3[1] + y0;
                        zi = 0.7328*R1[2] + 0.1461*R2[2] + 0.7855*R3[2] + z0;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 8, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        
                        // 7 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.5867*R1[0] + 0.8539*R2[0] + 0.2145*R3[0] + x0;
                        yi = 0.5867*R1[1] + 0.8539*R2[1] + 0.2145*R3[1] + y0;
                        zi = 0.5867*R1[2] + 0.8539*R2[2] + 0.2145*R3[2] + z0;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 8, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        
                        // 8 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.8539*R1[0] + 0.5867*R2[0] + 0.4521*R3[0] + x0;
                        yi = 0.8539*R1[1] + 0.5867*R2[1] + 0.4521*R3[1] + y0;
                        zi = 0.8539*R1[2] + 0.5867*R2[2] + 0.4521*R3[2] + z0;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 8, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        
                        // 9 atom of unit cell:
                        beginIndex = endIndex; endIndex = beginIndex + 4;
                        xi = 0.1461*R1[0] + 0.7328*R2[0] + 0.8812*R3[0] + x0;
                        yi = 0.1461*R1[1] + 0.7328*R2[1] + 0.8812*R3[1] + y0;;
                        zi = 0.1461*R1[2] + 0.7328*R2[2] + 0.8812*R3[2] + z0;;
                        distance = pow(x2-xi,2) + pow(y2-yi,2) + pow(z2-zi,2);
                        if (distance<rcut && distance>0.1) {ModifyNList(p,xi, yi, zi, 8, beginIndex, endIndex, atomNumber);}
                        atomNumber++;
                        nSize = nSize + 4*9;
                    } } }
                }
            }}//}
    //cout<<"The number of atoms considered is: "<<atomNumber<<endl;
}


void initializeCoordinates(PetscScalar* localXs, PetscScalar* localYs, PetscScalar* localZs, PetscInt *localAtoms,PetscInt* localBeginIndex, PetscInt* localEndIndex, PetscInt rbegin, PetscInt rend){
    //cout<<"Initializing coordinates:"<<endl;
    PetscInt atomNumber = 0;
    PetscScalar AngToBohr=1.88973;
    PetscScalar a =9.725; PetscScalar b=a; PetscScalar c=10.7; // in Bohrs
    PetscScalar R1[3], R2[3], R3[3];
    
    R1[0] = a;                  R1[1] = 0.0;            R1[2] = 0.0;
    R2[0] = b*(-sin(PI/6));     R2[1]= b*cos(PI/6);       R2[2] = 0.0;
    R3[0] = 0.0;                R3[1] = 0.0;            R3[2] = c;
    
    PetscScalar x0, y0, z0; PetscScalar xi, yi, zi;
    PetscInt localIndex = 0;
    PetscInt beginIndex=0; PetscInt endIndex=0;
    
    for (PetscInt i=0;i<L_lattice;i++){
        for (PetscInt j=0;j<M_lattice; j++){
            for (PetscInt k=0;k<N_lattice;k++){
                x0 = i*R1[0] + j*R2[0] + k*R3[0];
                y0 = i*R1[1] + j*R2[1] + k*R3[1];
                z0 = i*R1[2] + j*R2[2] + k*R3[2];
                
                // compute the positions of each atom:
                // 1 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.4697*R1[0] + 0.0000*R2[0] + 0.0000*R3[0];
                    yi = 0.4697*R1[1] + 0.0000*R2[1] + 0.0000*R3[1];
                    zi = 0.4697*R1[2] + 0.0000*R2[2] + 0.0000*R3[2];
                    
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localAtoms[localIndex] = 14;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localIndex++;
                }atomNumber++;
                
                // 2 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.0000*R1[0] + 0.4697*R2[0] + 0.6777*R3[0];
                    yi = 0.0000*R1[1] + 0.4697*R2[1] + 0.6777*R3[1];
                    zi = 0.0000*R1[2] + 0.4697*R2[2] + 0.6777*R3[2];
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localAtoms[localIndex] = 14;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localIndex++;
                }atomNumber = atomNumber + 1;
                
                // 3 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.5303*R1[0] + 0.5303*R2[0] + 0.3333*R3[0];
                    yi = 0.5303*R1[1] + 0.5303*R2[1] + 0.3333*R3[1];
                    zi = 0.5303*R1[2] + 0.5303*R2[2] + 0.3333*R3[2];
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localAtoms[localIndex] = 14;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localIndex++;
                }atomNumber = atomNumber + 1;
                
                // 4 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.4133*R1[0] + 0.2672*R2[0] + 0.1188*R3[0];
                    yi = 0.4133*R1[1] + 0.2672*R2[1] + 0.1188*R3[1];
                    zi = 0.4133*R1[2] + 0.2672*R2[2] + 0.1188*R3[2];
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localAtoms[localIndex] = 8;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localIndex++;
                }atomNumber = atomNumber + 1;
                
                // 5 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.2672*R1[0] + 0.4133*R2[0] + 0.5479*R3[0];
                    yi = 0.2672*R1[1] + 0.4133*R2[1] + 0.5479*R3[1];
                    zi = 0.2672*R1[2] + 0.4133*R2[2] + 0.5479*R3[2];
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localAtoms[localIndex] = 8;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localIndex++;
                }atomNumber = atomNumber + 1;
                
                
                // 6 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.7328*R1[0] + 0.1461*R2[0] + 0.7855*R3[0];
                    yi = 0.7328*R1[1] + 0.1461*R2[1] + 0.7855*R3[1];
                    zi = 0.7328*R1[2] + 0.1461*R2[2] + 0.7855*R3[2];
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localAtoms[localIndex] = 8;
                    localIndex++;
                }atomNumber = atomNumber + 1;
                
                // 7 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.5867*R1[0] + 0.8539*R2[0] + 0.2145*R3[0];
                    yi = 0.5867*R1[1] + 0.8539*R2[1] + 0.2145*R3[1];
                    zi = 0.5867*R1[2] + 0.8539*R2[2] + 0.2145*R3[2];
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localAtoms[localIndex] = 8;
                    localIndex++;
                }atomNumber = atomNumber + 1;
                
                // 8 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.8539*R1[0] + 0.5867*R2[0] + 0.4521*R3[0];
                    yi = 0.8539*R1[1] + 0.5867*R2[1] + 0.4521*R3[1];
                    zi = 0.8539*R1[2] + 0.5867*R2[2] + 0.4521*R3[2];
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localAtoms[localIndex] = 8;
                    localIndex++;
                }atomNumber = atomNumber + 1;
                
                // 9 atom of unit cell:
                beginIndex = endIndex; endIndex = beginIndex + 4;
                if (atomNumber>=rbegin && atomNumber<rend){
                    xi = 0.1461*R1[0] + 0.7328*R2[0] + 0.8812*R3[0];
                    yi = 0.1461*R1[1] + 0.7328*R2[1] + 0.8812*R3[1];
                    zi = 0.1461*R1[2] + 0.7328*R2[2] + 0.8812*R3[2];
                    localXs[localIndex] = x0 + xi;
                    localYs[localIndex] = y0 + yi;
                    localZs[localIndex] = z0 + zi;
                    localBeginIndex[localIndex] = beginIndex;
                    localEndIndex[localIndex] = endIndex;
                    localAtoms[localIndex] = 8;
                    localIndex++;
                }atomNumber = atomNumber + 1;
                
            }
        }
    }
}

int arrangeCoordinates(){
    // create global petsc vectors for atoms;
    ierr = VecCreate(PETSC_COMM_WORLD,&Xs);CHKERRQ(ierr);
    ierr = VecSetSizes(Xs,PETSC_DECIDE,Natoms);CHKERRQ(ierr);
    ierr = VecSetFromOptions(Xs);CHKERRQ(ierr);
    ierr = VecDuplicate(Xs,&Ys);CHKERRQ(ierr);
    ierr = VecDuplicate(Xs,&Zs);CHKERRQ(ierr);
    //ierr = VecDuplicate(Xs, &netCharges); CHKERRQ(ierr);
    //ierr = VecDuplicate(Xs, &gammaq); CHKERRQ(ierr);
    //ierr = VecDuplicate(Xs, &neutralCharges); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(Xs,&rbegin,&rend);CHKERRQ(ierr);
    
    
    // (a) initialize local data on atoms:
    ni = rend - rbegin;
    //cout<<" ni is : "<<ni<<endl;
    localXs = new PetscScalar[ni];
    localYs = new PetscScalar[ni];
    localZs = new PetscScalar[ni];
    localAtoms = new PetscInt[ni];
    localBeginIndex = new PetscInt[ni];       // begin index of this local atom on the global Hamiltonian
    localEndIndex = new PetscInt[ni];
    //localnetCharges = new PetscScalar[ni];
    //localneutralCharges = new PetscScalar[ni];
    //localgammaq = new PetscScalar[ni];
    
    // (b) initialize local data on neighbors:
    nN = new PetscInt[ni];
    xsN = new PetscScalar[ni*maxN_A];
    ysN = new PetscScalar[ni*maxN_A];
    zsN = new PetscScalar[ni*maxN_A];
    aTN = new PetscInt[ni*maxN_A]; // atomtypeNeighbor
    bIN = new PetscInt[ni*maxN_A]; // beginIndexNeighbor
    eIN = new PetscInt[ni*maxN_A]; // endIndexNeighbor
    gAIN = new PetscInt[ni*maxN_A]; // global Atom Index Neighbor
    //ito = new PetscInt[ni*maxN_A];
    
    for (int i=0;i<ni;i++){nN[i]=0; }//localnetCharges[i]=0.0; localgammaq[i]=0.0; localneutralCharges[i]=0.0;}
    
    initializeCoordinates(localXs, localYs, localZs, localAtoms, localBeginIndex, localEndIndex, rbegin, rend);
    getHaloAtoms(ni, rbegin, rend, localXs, localYs, localZs);
    
    /*for (int i=0; i<ni;i++){
        if (localAtoms[i]==18){
            localneutralCharges[i]=4;
        }else if (localAtoms[i]==14){
            localneutralCharges[i]=2;
        }
    }
    
    ierr = VecCreateSeq(PETSC_COMM_SELF, ni*maxN_A,&netQNeighbors); CHKERRQ(ierr);
    ierr = VecSetFromOptions(netQNeighbors); CHKERRQ(ierr);
    */
    PetscInt idx[ni];
    for (PetscInt i=0;i<(rend-rbegin);i++){
        idx[i] = rbegin + i;
    }
    ierr = VecSetValues(Xs,ni,idx, localXs ,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValues(Ys,ni,idx, localYs ,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValues(Zs,ni,idx, localZs ,INSERT_VALUES); CHKERRQ(ierr);
    
    VecAssemblyBegin(Xs); VecAssemblyEnd(Xs);
    VecAssemblyBegin(Ys); VecAssemblyEnd(Ys);
    VecAssemblyBegin(Zs); VecAssemblyEnd(Zs);
    return 0;
}

int clearCoordinateInformation(){
    if (localXs!=NULL){delete [] localXs; localXs = NULL;}
    if (localYs!=NULL){delete [] localYs; localYs = NULL;}
    if (localZs!=NULL){delete [] localZs; localZs = NULL;}
    if (localAtoms!=NULL){delete [] localAtoms; localAtoms = NULL; }
    if (localBeginIndex!=NULL){delete [] localBeginIndex; localBeginIndex=NULL;}
    if (localEndIndex!=NULL){delete [] localEndIndex; localEndIndex=NULL;}
    
    
    // (b) initialize local data on neighbors:
    if (nN!=NULL){delete [] nN; nN = NULL;}
    if (xsN!=NULL){delete [] xsN; xsN = NULL;}
    if (ysN!=NULL){delete [] ysN; ysN = NULL;}
    if (zsN!=NULL){delete [] zsN; zsN = NULL;}
    if (aTN!=NULL){delete [] aTN; aTN = NULL;} // atomtypeNeighbor
    if (bIN!=NULL){delete [] bIN; bIN = NULL;} // beginIndexNeighbor
    if (eIN!=NULL){delete [] eIN; eIN = NULL;} // endIndexNeighbor
    if (gAIN!=NULL){delete [] gAIN; gAIN = NULL;} // global Atom Index Neighbor
    
    ierr = VecDestroy(&Xs); CHKERRQ(ierr);
    ierr = VecDestroy(&Ys); CHKERRQ(ierr);
    ierr = VecDestroy(&Zs); CHKERRQ(ierr); return 0;
}

int initializeForH(){
    //! Creates the matrix/preallocates but does not set values:
    //! is called only once
    //! vectors allocated: diagonalH0G, offDiagonalH0G, diagonaldHG, offDiagonaldHG, diagP:
    
    ierr = VecCreate(PETSC_COMM_WORLD,&diagonalH0G); CHKERRQ(ierr);
    ierr = VecSetSizes(diagonalH0G,PETSC_DECIDE,nSize);CHKERRQ(ierr);
    ierr = VecSetFromOptions(diagonalH0G);CHKERRQ(ierr);
    ierr = VecDuplicate(diagonalH0G, &offDiagonalH0G); CHKERRQ(ierr);
    ierr = VecDuplicate(diagonalH0G, &diagonaldHG); CHKERRQ(ierr);
    ierr = VecDuplicate(diagonalH0G, &offDigonaldHG); CHKERRQ(ierr);
    ierr = VecDuplicate(diagonalH0G, &diagP);  CHKERRQ(ierr);

    ierr = VecDuplicate(diagonalH0G, &LambdaMaxs);  CHKERRQ(ierr);
    ierr = VecDuplicate(diagonalH0G, &LambdaMins);  CHKERRQ(ierr); 
    // set up and create the Hamiltonian matrix, and rescaled matrix:
    // matrices allocated: H, deltaH, Htotal;
    ierr = MatCreate(PETSC_COMM_WORLD,&H); CHKERRQ(ierr); ierr=MatSetSizes(H,PETSC_DECIDE,PETSC_DECIDE,nSize,nSize); CHKERRQ(ierr);
    ierr = MatSetFromOptions(H); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(H, maxN_A*5,NULL); CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(H, maxN_A*5,NULL,maxN_A*5,NULL); CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&Htotal); CHKERRQ(ierr); ierr=MatSetSizes(Htotal,PETSC_DECIDE,PETSC_DECIDE,nSize,nSize); CHKERRQ(ierr);
    ierr = MatSetFromOptions(Htotal); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Htotal, maxN_A*5,NULL); CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(Htotal, maxN_A*5,NULL,maxN_A*5,NULL); CHKERRQ(ierr);
    return 0;
}

int writeCoordFile(){
    //! print out the atomnumber, atomtype, positions and neighbor list, mostly for debugging or plotting purposes:
    PetscInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    if (rank==0){
        FILE* coordfile = fopen("coordfile","w");
        for (int i=0; i<(rend - rbegin); i++){
            fprintf(coordfile,"%d  %d  %lf  %lf %lf ", i, localAtoms[i], localXs[i], localYs[i], localZs[i]);
            for (int j=0; j<nN[i]; j++){
                fprintf(coordfile,"%d  ", gAIN[i*maxN_A + j]);
            }
            fprintf(coordfile," \n");
        }
        fclose(coordfile);
    }
    return 0;
}

int constructH(){
    // local storage required for inputting values to hvalues-matrix:
    PetscInt globalRow;
    PetscInt globalCol;
    PetscScalar* hvalues = new PetscScalar[16];
    for (int ii=0; ii<16;ii++){ hvalues[ii]=0.0;}
    
    //FILE* hmatrix = fopen("hmatrix","w");
    for (PetscInt i=0;i<ni; i++){
        PetscScalar xi = localXs[i]; PetscScalar yi = localYs[i]; PetscScalar zi = localZs[i];
        // get diagonal hvalues:
        get4by4HmatrixElements(localAtoms[i], localAtoms[i], 0.0, 0.0, 0.0, hvalues);
        // get indices for diagonal hvalues:
        globalRow = localBeginIndex[i]; globalCol = localBeginIndex[i];
        
        for (PetscInt i1=0; i1<4; i1++){
            PetscInt cols[4];
            PetscScalar vals[4];
            PetscInt row = globalRow + i1;
            
            for (int j1=0; j1<4; j1++) {
                cols[j1] = globalCol + j1; vals[j1] = hvalues[i1*4+j1];
                //fprintf(hmatrix,"%d %d  %lf \n", row+1, cols[j1]+1, vals[j1]);
            } //cout<<row<<"  "<<cols[j1]<<"    "<<vals[j1]<<endl;}
            ierr = MatSetValues(H, 1,&row,4,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValues(Htotal, 1,&row,4,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
        }
        
        PetscScalar diagonalH[4];
        PetscScalar offDiagonalH[4];
        for (int q=0; q<4; q++){
            diagonalH[q] = hvalues[q*4+q];
            offDiagonalH[q] = 0.0;
        }
        
        // get values associated with neighbor atoms:
        for (PetscInt j=0;j<nN[i];j++){
            PetscScalar xj = xsN[i*maxN_A + j];
            PetscScalar yj = ysN[i*maxN_A + j];
            PetscScalar zj = zsN[i*maxN_A + j];
            PetscInt aT = aTN[i*maxN_A + j];
            globalCol = bIN[i*maxN_A + j];
            
            get4by4HmatrixElements(localAtoms[i], aT, (xj-xi), (yj-yi), (zj-zi), hvalues);
            
            for (PetscInt i1=0; i1<4; i1++){
                PetscInt cols[4];
                PetscScalar vals[4];
                PetscInt row = globalRow + i1;
                
                for (int j1=0; j1<4; j1++) {
                    cols[j1] = globalCol + j1; vals[j1] = hvalues[i1*4+j1];
                }
                ierr = MatSetValues(H, 1,&row,4,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValues(Htotal, 1,&row,4,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
            }
            
            for (int q=0;q<4;q++){
                PetscScalar val = 0.0;
                for (int qq=0;qq<4; qq++){
                    val = val + abs(hvalues[q*4 + qq]);
                }
                offDiagonalH[q] = offDiagonalH[q] + val;
            }
        }
        
        PetscScalar forLambdaMin[4];
        PetscScalar forLambdaMax[4];
        PetscInt Rows[4];
        for (int q=0;q<4;q++){
            Rows[q] = localBeginIndex[i] + q;
            //forLambdaMin[q] = diagonalH[q] - offDiagonalH[q];
            //forLambdaMax[q] = diagonalH[q] + offDiagonalH[q];
        }
        ierr = VecSetValues(diagonalH0G,4,Rows,diagonalH,INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecSetValues(offDiagonalH0G,4,Rows,offDiagonalH,INSERT_VALUES); CHKERRQ(ierr);
    }
    //fclose(hmatrix);
    ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);   CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Htotal,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Htotal,MAT_FINAL_ASSEMBLY);   CHKERRQ(ierr);
    
    //ierr = MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, deltaH); CHKERRQ(ierr);
    //ierr = MatDuplicate(H, MAT_COPY_VALUES, &Htotal); CHKERRQ(ierr);
    
    // ierr = MatView(H,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(diagonalH0G);  CHKERRQ(ierr);
    ierr = VecAssemblyEnd(diagonalH0G);    CHKERRQ(ierr);
    
    ierr = VecAssemblyBegin(offDiagonalH0G);  CHKERRQ(ierr);
    ierr = VecAssemblyEnd(offDiagonalH0G);    CHKERRQ(ierr);
    
    ierr = VecWAXPY(LambdaMaxs, 1.0, offDiagonalH0G, diagonalH0G); CHKERRQ(ierr);
    ierr = VecWAXPY(LambdaMins, -1.0, offDiagonalH0G, diagonalH0G); CHKERRQ(ierr);
    return 0;
}

int getKPE(Vec inVector, Vec outVector){
    Vec randomVector, X0RV, x_i, x_m1, x_p1, Hx_i, thetax;
    ierr = VecCreate(PETSC_COMM_WORLD,&randomVector); CHKERRQ(ierr);
    ierr = VecSetSizes(randomVector,PETSC_DECIDE,nSize);CHKERRQ(ierr);
    ierr = VecSetFromOptions(randomVector);CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &X0RV); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &x_i); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &x_m1); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &x_p1); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &Hx_i); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &thetax); CHKERRQ(ierr);
    
    
    ierr = VecCopy(inVector, randomVector); CHKERRQ(ierr);
    
    ierr = VecSet(X0RV,0); CHKERRQ(ierr);
    ierr = VecSet(thetax,0); CHKERRQ(ierr);
    ierr = VecSet(x_i,0); CHKERRQ(ierr);
    ierr = VecSet(x_m1,0); CHKERRQ(ierr);
    ierr = VecSet(x_p1,0); CHKERRQ(ierr);
    ierr = VecSet(Hx_i,0); CHKERRQ(ierr);
    
    double alpha = 1.0; double beta = 0.0;
    
    // getMoments :
    //gamma = muval;
    int M = 200;
    double* mu_mvals = new double[M];
    double* Uis = new double[M];
    double* a_i_M = new double[M];
    double* gmMs = new double[M];
    //muval = 0.95; 
    mu_mvals[0] = 2*(1 - acos(muval)/PI);
    for (int m=1; m<M; m++){
        mu_mvals[m]=-2*sin(m*acos(muval))/(m*PI);
    }
    
    double alpha_M = (PI/(M+2)); // since the max index=M-1
    double lambda_max_M = cos(alpha_M);
    
    double Um1 = 0.0; double normOfUis = 0.0;
    for (int m=0; m<M; m++) {
        Uis[m]=sin((m+1)*acos(lambda_max_M))/sin(acos(lambda_max_M));
        normOfUis = normOfUis + pow(Uis[m],2);
    }
     normOfUis = sqrt(normOfUis);
    /* Replacing BLAS1 functions with for-loops
    double normOfUis = cblas_dnrm2(M,Uis,1);
    cblas_dcopy(M, Uis,1, a_i_M,1);
    cblas_dscal(M, 1./normOfUis, a_i_M, 1);
    */
    for (int i=0;i<M;i++){
        a_i_M[i] = Uis[i]/normOfUis;
    }
    

    double gmMval = 0.0;
    
    for (int m=0;m<M;m++){
        gmMval = 0.0;
        for (int i=0; i<(M-m-1); i++){
            gmMval = gmMval + a_i_M[i]*a_i_M[m+i];
        }
        gmMs[m] = gmMval;
    }
    // zeroth iteration:
    ierr = VecCopy(randomVector,x_i); CHKERRQ(ierr);
    
    ierr = VecAXPY(thetax, (mu_mvals[0]/2.0), x_i); CHKERRQ(ierr);
    
    // 1-th iteration:
    ierr = MatMult(Htotal, x_i, Hx_i); CHKERRQ(ierr);
    ierr = VecCopy(x_i, x_m1); CHKERRQ(ierr);
    ierr = VecCopy(Hx_i, x_i); CHKERRQ(ierr);
    ierr = VecAXPY(thetax, (mu_mvals[1]*gmMs[1]), x_i); CHKERRQ(ierr);
    
    
    //if (debugHS==1) CkPrintf("The other values are %lf and %lf \n", mu_mvals[1], gmMs[1]);
    
    // from 2-th iteration onwards: the Chebyshev iteration relation holds:
    for (int i=2;i<M;i++){
        ierr = MatMult(Htotal, x_i, Hx_i); CHKERRQ(ierr);
        //mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, X, descrA, x_i, beta, Hx_i);
        ierr = VecCopy(Hx_i, x_p1);  CHKERRQ(ierr);               // cblas_dcopy(nSize, Hx_i, 1, x_p1,1);
        
        
        ierr = VecScale(x_p1, 2.0);  CHKERRQ(ierr);               // cblas_dscal(nSize, 2.0, x_p1, 1);
        
        
        ierr = VecAXPY(x_p1, -1.0, x_m1);           // cblas_daxpy(nSize,-1.0, x_m1, 1, x_p1,1);
        CHKERRQ(ierr);
        
        ierr = VecCopy(x_i, x_m1);                  // cblas_dcopy(nSize, x_i,1,x_m1,1);
        CHKERRQ(ierr);
        
        ierr = VecCopy(x_p1, x_i);                  // cblas_dcopy(nSize, x_p1, 1, x_i, 1);
        CHKERRQ(ierr);
        
        ierr = VecAXPY(thetax, (mu_mvals[i]*gmMs[i]), x_i); //cblas_daxpy(nSize, (mu_mvals[i]*gmMs[i]), x_i,1,thetax,1);
        CHKERRQ(ierr);
    }
    
    VecCopy(thetax, outVector);
    VecScale(outVector, 2.0);
 
    VecDestroy(&randomVector);
    VecDestroy(&X0RV);
    VecDestroy(&x_i);
    VecDestroy(&x_m1);
    VecDestroy(&x_p1);
    VecDestroy(&Hx_i);
    VecDestroy(&thetax);
    
    delete [] mu_mvals;
    delete [] Uis;
    delete [] a_i_M;
    delete [] gmMs;
    return 0;
}

int getSP2(Vec vectorIn, Vec vectorOut, int localsp2max, int currentsp2){
    Vec x, y, z;
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, nSize); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &y); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &z); CHKERRQ(ierr);
    
    ierr = VecCopy(vectorIn, x); CHKERRQ(ierr);
    
    if (currentsp2==0){
        getKPE(x,y);
        ierr = VecCopy(y, vectorOut); CHKERRQ(ierr);
    } else if (currentsp2>0){
        getSP2(x,y,localsp2max,currentsp2-1);
        getSP2(y,z,localsp2max,currentsp2-1);
        PetscScalar alphacurrent = alphavals[currentsp2-1];
        VecCopy(y,vectorOut);                  // vectorOut = y
        VecScale(vectorOut, 1.0 + alphacurrent); // vectorOut = (1+alphacurrent)*y
        VecAXPY(vectorOut, -alphacurrent, z);    // vectorOut = (a+alphacurrent)*y - alphacurrent*z
        
    } else if (currentsp2 < 0){
        cout<<"some error with calling sp2 :: negative depth"<<endl;
    }
    /*ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&y); CHKERRQ(ierr);
    ierr = VecDestroy(&z); CHKERRQ(ierr);
   */ return 0;
}

int computeAlphas(){
    //CkPrintf("maxsp2 and nRandoms : %d %d , M is %d \n", maxsp2, nRandoms, M);
    // double tstart = clock();
    Vec         rvector, sp2Input, sp2Output, v2;
    PetscScalar placeholder;
    //PetscInt     MKPE = 100;
    //PetscScalar* momentsTrace = new PetscScalar[MKPE];
    
    ierr = VecCreate(PETSC_COMM_WORLD, &rvector); CHKERRQ(ierr);
    ierr = VecSetSizes(rvector, PETSC_DECIDE, nSize); CHKERRQ(ierr);
    ierr = VecSetFromOptions(rvector); CHKERRQ(ierr);
    ierr = VecDuplicate(rvector, &sp2Input); CHKERRQ(ierr);
    ierr = VecDuplicate(rvector, &sp2Output); CHKERRQ(ierr);
    ierr = VecDuplicate(rvector, &v2); CHKERRQ(ierr);
    
    PetscRandom    rctx;
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
    PetscRandomSetFromOptions(rctx);
     double trace_sp2 = 0.0;

    for (int sp2count=0; sp2count<maxsp2; sp2count++){
        
        trace_sp2 = 0.0;
        for (int randomcount=0;randomcount<nRandoms;randomcount++){
            VecSetRandom(rvector,rctx);
	    VecAssemblyBegin(rvector); VecAssemblyEnd(rvector);
	    VecShift(rvector,-0.5);

            VecCopy(rvector, sp2Input);
            
            getSP2(sp2Input, sp2Output, sp2count, sp2count);

            VecAXPY(sp2Input,  -1.0, sp2Output);
            PetscScalar diffval;
            VecNorm(sp2Input, NORM_2,&diffval); //cout<<" diff : "<<(double)diffval<<endl;

            PetscScalar placeholder = 0.0;
            VecDot(rvector, sp2Output,&placeholder);
            trace_sp2 = trace_sp2 + placeholder/nRandoms;
        }
        if (MPI_RANK==0) printf("sp2count is %d, and trace is %lf \n",sp2count,trace_sp2);
        if (fabs(trace_sp2*2 - Nelectrons)<0.001) break;
        
        if (trace_sp2*2> Nelectrons) alphavals[sp2count] = -1.0;
        else alphavals[sp2count] = 1.0;
    }
    // double tend = clock();
    // printf("Time taken for SP2 steps : %lf \n",(tend - tstart));
    cout<<" The number of electrons is:"<< Nelectrons<<endl;
    cout<<"The obtained trace at end is: "<< trace_sp2<<endl;
 
    ierr = VecDestroy(&rvector); CHKERRQ(ierr);
    ierr = VecDestroy(&sp2Input); CHKERRQ(ierr);
    ierr = VecDestroy(&sp2Output); CHKERRQ(ierr);
    ierr = VecDestroy(&v2); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);
    //delete [] momentsTrace; momentsTrace = NULL;
    return 0;
}

int computeDiagonalElements(){
    Vec randomVector, numeratorVals, denominatorVals, sp2Input, sp2Output, placeholder;
    //! uses computed alpha values to modify net Charges:
    ierr = VecCreate(PETSC_COMM_WORLD, &randomVector); CHKERRQ(ierr);
    ierr = VecSetSizes(randomVector, PETSC_DECIDE, nSize); CHKERRQ(ierr);
    ierr = VecSetFromOptions(randomVector); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &numeratorVals); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &denominatorVals); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &sp2Input); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &sp2Output); CHKERRQ(ierr);
    ierr = VecDuplicate(randomVector, &placeholder); CHKERRQ(ierr);
    
    PetscRandom    rctx;
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
    PetscRandomSetFromOptions(rctx);
    
    double meanvalue = 0.0;
    double sigmavalue = 1.0;
    VecSet(numeratorVals,0.0);
    VecSet(denominatorVals,0.0);
    
    
    double alpha = 1.0; double beta = 0.0;
    
    for (int randomcount=0;randomcount<nRandoms;randomcount++){
        VecSetRandom(randomVector, rctx);//vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, nSize, randomVector, meanvalue, sigmavalue);
        VecShift(randomVector, -0.5);
	PetscScalar normval;
	//VecView(randomVector, PETSC_VIEWER_STDOUT_WORLD);
	VecNorm(randomVector, NORM_2, &normval);
        VecScale(randomVector, 1.0/normval);

        VecCopy(randomVector, sp2Input); //cblas_dcopy(nSize, randomVector,1, sp2Input,1);
        getSP2(sp2Input, sp2Output, maxsp2-1, maxsp2-1);
        
        VecPointwiseMult(placeholder, sp2Input, sp2Output);
        VecAXPY(numeratorVals, 1.0, placeholder);
        
        VecPointwiseMult(placeholder, sp2Input, sp2Input);
        VecAXPY(denominatorVals, 1.0, placeholder);
       
        VecAXPY(sp2Input,  -1.0, sp2Output);
        PetscScalar diffval;
        VecNorm(sp2Input, NORM_2,&diffval); //cout<<" diff : "<<(double)diffval<<endl;
    }
    
    ierr = VecPointwiseDivide(diagP, numeratorVals, denominatorVals); CHKERRQ(ierr);
    //ierr = VecWAXPY(netCharges, -1.0, neutralCharges, diagP); CHKERRQ(ierr);
    //VecView(diagP, PETSC_VIEWER_STDOUT_WORLD);
    PetscScalar tempN;
    VecNorm(diagP, NORM_1, &tempN);
    cout<<" The sum of values is : "<< tempN<<endl;

    ierr = VecDestroy(&randomVector); CHKERRQ(ierr);
    ierr = VecDestroy(&sp2Input); CHKERRQ(ierr);
    ierr = VecDestroy(&sp2Output); CHKERRQ(ierr);
    ierr = VecDestroy(&placeholder); CHKERRQ(ierr);
    ierr = VecDestroy(&numeratorVals); CHKERRQ(ierr);
    ierr = VecDestroy(&denominatorVals); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);
    return 0;
}

int computeDensityMatrix(){
    maxsp2 = 5;
    alphavals = new double[maxsp2];
    nRandoms = 10;
    for (int i=0;i<maxsp2;i++) alphavals[i]=0.0;
    
    computeAlphas();
    //for (int i=0;i<maxsp2;i++) cout<<" alphval is :"<<alphavals[i]<<endl;

    computeDiagonalElements();
    delete [] alphavals;
    return 0;
}


int constructdeltaH(){
    /*
    //obtainGammaQ();
    PetscInt globalAtomIndex;
    PetscInt globalRow;
    PetscInt globalCol;
    PetscScalar* hvalues = new PetscScalar[16];
    for (int ii=0; ii<16;ii++){ hvalues[ii]=0.0;}
    
    //FILE* hmatrix = fopen("hmatrix","w");
    for (PetscInt i=0;i<ni; i++){
        globalAtomIndex = rbegin + i;
        // get diagonal hvalues:
        get4by4deltaH(globalAtomIndex, globalAtomIndex, 0.0, 0.0, 0.0, hvalues);
        // get indices for diagonal hvalues:
        globalRow = localBeginIndex[i]; globalCol = localBeginIndex[i];
        
        for (PetscInt i1=0; i1<4; i1++){
            PetscInt cols[4];
            PetscScalar vals[4];
            PetscInt row = globalRow + i1;
            
            for (int j1=0; j1<4; j1++) {
                cols[j1] = globalCol + j1; vals[j1] = hvalues[i1*4+j1];
                //fprintf(hmatrix,"%d %d  %lf \n", row+1, cols[j1]+1, vals[j1]);
            } //cout<<row<<"  "<<cols[j1]<<"    "<<vals[j1]<<endl;}
            ierr = MatSetValues(H, 1,&row,4,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
        }
        
        PetscScalar diagonalH[4];
        PetscScalar offDiagonalH[4];
        for (int q=0; q<4; q++){
            diagonalH[q] = hvalues[q*4+q];
            offDiagonalH[q] = 0.0;
        }
        
        // get values associated with neighbor atoms:
        for (PetscInt j=0;j<nN[i];j++){
            PetscScalar xj = xsN[i*maxN_A + j];
            PetscScalar yj = ysN[i*maxN_A + j];
            PetscScalar zj = zsN[i*maxN_A + j];
            PetscInt aT = aTN[i*maxN_A + j];
            globalCol = bIN[i*maxN_A + j];
            
            get4by4HmatrixElements(localAtoms[i], aT, (xj-xi), (yj-yi), (zj-zi), hvalues);
            
            for (PetscInt i1=0; i1<4; i1++){
                PetscInt cols[4];
                PetscScalar vals[4];
                PetscInt row = globalRow + i1;
                
                for (int j1=0; j1<4; j1++) {
                    cols[j1] = globalCol + j1; vals[j1] = hvalues[i1*4+j1];
                }
                ierr = MatSetValues(H, 1,&row,4,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
            }
            
            for (int q=0;q<4;q++){
                PetscScalar val = 0.0;
                for (int qq=0;qq<4; qq++){
                    val = val + abs(hvalues[q*4 + qq]);
                }
                offDiagonalH[q] = offDiagonalH[q] + val;
            }
        }
        
        PetscScalar forLambdaMin[4];
        PetscScalar forLambdaMax[4];
        PetscInt Rows[4];
        for (int q=0;q<4;q++){
            Rows[q] = localBeginIndex[i] + q;
            //forLambdaMin[q] = diagonalH[q] - offDiagonalH[q];
            //forLambdaMax[q] = diagonalH[q] + offDiagonalH[q];
        }
        ierr = VecSetValues(diagonalH0G,4,Rows,diagonalH,INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecSetValues(offDiagonalH0G,4,Rows,offDiagonalH,INSERT_VALUES); CHKERRQ(ierr);
    }*/
    return 0;
}

int runKPM(){
    
    PetscScalar lambda_max, lambda_min;
    ierr = VecMax(LambdaMaxs, NULL, &lambda_max); CHKERRQ(ierr);
    ierr = VecMin(LambdaMins, NULL, &lambda_min); CHKERRQ(ierr);
    
    PetscScalar xmax;
    ierr = VecMax(Xs,NULL,&xmax); CHKERRQ(ierr);
    //cout<<"the max along x is : "<<xmax<<endl;
    
    PetscScalar a = (lambda_max - lambda_min)/2;
    PetscScalar b = (lambda_max + lambda_min)/2;
    cout<<" lambda_max is : "<<lambda_max<<"lambda_min is "<<lambda_min<<endl;
    cout<<" a is: "<<a<<"  b is : "<<b<<endl;
    
    ierr = MatShift(Htotal, -b);  CHKERRQ(ierr);      // H --> (H - b)/a
    ierr = MatScale(Htotal, 1.0/a); CHKERRQ(ierr);
    // ierr = MatView(H,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    
    // Compute the density of states:
    Vec         rvector, v0, v1, v2;
    PetscInt    Nmoments = 1000;
    PetscScalar* momentsTrace = new PetscScalar[Nmoments];
    
    ierr = VecCreate(PETSC_COMM_WORLD, &rvector); CHKERRQ(ierr);
    ierr = VecSetSizes(rvector, PETSC_DECIDE, nSize); CHKERRQ(ierr);
    ierr = VecSetFromOptions(rvector); CHKERRQ(ierr);
    ierr = VecDuplicate(rvector, &v0); CHKERRQ(ierr);
    ierr = VecDuplicate(rvector, &v1); CHKERRQ(ierr);
    ierr = VecDuplicate(rvector, &v2); CHKERRQ(ierr);
    
    PetscRandom    rctx;
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
    PetscRandomSetFromOptions(rctx);
    
    // Kernel Polynomial Method:
    PetscInt nTilde = 2*Nmoments;
    vector<PetscScalar> gns(Nmoments);
    vector<PetscScalar> xk(nTilde);
    vector<PetscScalar> DOSvals(nTilde);
    
    for (PetscInt i=0;i<Nmoments; i++) momentsTrace[i]=0.0;
    PetscInt nRandom = 20;
    PetscScalar* Energies = new PetscScalar[nTilde];
    // Obtain xk:
    for (PetscInt k=0; k<nTilde; k++){
        xk[k] = cos(PI*(k+1.0/2.0)/(nTilde));
    }
    // getJacksonKernel(nMoments);
    for (PetscInt n=0; n<Nmoments; n++){
        gns[n] = ((Nmoments - n + 1)*cos(PI*PetscScalar(n)/PetscScalar(Nmoments+1)) + sin(PI*PetscScalar(n)/PetscScalar(Nmoments+1))*cot(PI/PetscScalar(Nmoments+1)))/PetscScalar(Nmoments+1);
    }
    
    // obtain Chebyshev moments
    PetscScalar traceval = 0.0;
    PetscScalar placeholder=0.0;
    for (PetscInt i=0;i<nRandom;i++){
        VecSetRandom(rvector,rctx);
        VecAssemblyBegin(rvector); VecAssemblyEnd(rvector);
	VecShift(rvector, -0.5);
        VecCopy(rvector, v0);
        VecDot(rvector, v0, &placeholder);        // matrix multiply Tminus1*randomVector;
        momentsTrace[0] = momentsTrace[0] + placeholder;
        
        MatMult(Htotal, v0,v1);                   // matrix vector multiply X*randomVector;
        VecDot(rvector, v1, &placeholder);
        momentsTrace[1] = momentsTrace[1] + placeholder;
        
        for (PetscInt i=2; i<Nmoments; i++){
            MatMult(Htotal, v1,v2);    // goal: v2 = 2*X*v1 - x0 :
            VecScale(v2, 2.0);
            VecAXPY(v2, -1.0, v0);
            VecDot(rvector, v2, &placeholder);
            momentsTrace[i] = momentsTrace[i] + placeholder;
            VecCopy(v1 , v0);   // v0 = v1
            VecCopy(v2 , v1);   // v1 = v2;
        }
    }
    // Obtain DOS between [-1,1]
    PetscScalar xvalue = 0.0; PetscScalar dosvalue = 0.0;
    
    for (PetscInt i=0;i<nTilde;i++){
        xvalue = xk[i];
        dosvalue = gns[0]*momentsTrace[0];
        for (PetscInt j=1;j<Nmoments;j++){
            dosvalue = dosvalue + 2*gns[j]*momentsTrace[j]*cos(j*acos(xvalue));
        }
        dosvalue = dosvalue/(PI*sqrt(1.0-pow(xvalue,2)));
        DOSvals[i]=dosvalue;
    }
    // Get the chemical potential mu_1 according to the dos-estimate, and find mu_2
    // also find better max. and min. eigenvalues:
    
    // Obtain the chemical potential E [-1,1] :
    // Unscale the Energy values:
    
    for (int i=0; i<nTilde; i++){
        Energies[i] = xk[i]*a + b;
    }

    //FILE* file1 = fopen("dosre.txt","w"); 
    // Integrate for total area under DOS curve:
    double dosall = 0.0;
    for (int i=(nTilde-1);i>0;i--){
        dosall=dosall + (xk[i-1]-xk[i])*(DOSvals[i] + DOSvals[i-1])/2;
      //  fprintf(file1," %lf   %lf  \n", xk[i], DOSvals[i]);
    }
    //fclose(file1);
    
    // determine fraction of area:
    double areaFraction = 2.0/3.0*dosall;//(double)Nelectrons*dosall/(2*nSize);
    cout<< "dosall is "<<dosall<<"  areaFraction is :"<<areaFraction<<endl;

    // determine the energy that gives this value in correct order:
    double dos_mu = 0.0;
    int muIndex = 0;
    
    for (int i=1;i<nTilde;i++){
        dos_mu=dos_mu + (xk[i-1]-xk[i])*(DOSvals[i] + DOSvals[i-1])/2;
        //if (debugHS==1) cout<<xk[i]<<endl;
        if (dos_mu>=(areaFraction)) {
            muIndex = i;
            break;
        }
    }
    double chemicalPotential = Energies[muIndex];
    /*double minEnergyguess = -10;
    double maxEnergyguess= 10;
    
    for (int i=(nTilde-1);i>0;i--){
        if ((DOSvals[i]/nSize)>=0.001){
            minEnergyguess = Energies[i];
            break;
        }
    }
    
    for (int i=0;i<nTilde;i++){
        if ((DOSvals[i]/nSize)>=0.001){
            maxEnergyguess = Energies[i];
            break;
        }
    }
    cout<<"Min energy "<<minEnergyguess<<"  maxenergy: "<<maxEnergyguess<<endl;

    ierr = MatScale(Htotal, a); CHKERRQ(ierr);
    ierr = MatShift(Htotal, b); CHKERRQ(ierr);
    
    a =  (maxEnergyguess - minEnergyguess)/(2);
    b = (maxEnergyguess + minEnergyguess)/2;
    
    ierr = MatShift(Htotal, -b); CHKERRQ(ierr);
    ierr = MatScale(Htotal, 1.0/a); CHKERRQ(ierr);
    //rescaleH(maxeig, mineig);
    */
    muval = (chemicalPotential - b)/a;
    cout<<"muval is :"<<muval<<endl;
    return 0;
}

int initializeDiagP(){
    PetscScalar* localDiagP = new PetscScalar[localHSize];
    PetscInt* Rows = new PetscInt[localHSize];
    PetscScalar* perAtomP = new PetscScalar[localHSize];
    
    PetscInt row0 = localBeginIndex[0];
    for (PetscInt i=0;i<localHSize; i++){
        Rows[i] = row0 + i;
        perAtomP[i] = 0.0;
    }
    ierr = VecSetValues(diagP,localHSize,Rows,perAtomP,INSERT_VALUES); CHKERRQ(ierr);
    VecAssemblyBegin(diagP); VecAssemblyEnd(diagP);
    delete [] localDiagP; 
    delete [] Rows;
    delete [] perAtomP;
    return 0;
}

PetscInt main(PetscInt argc, char** argv){
    // assume alpha-quartz:
    M_lattice = 1; L_lattice = 1; N_lattice = 1;
    if (argc>=1){
        string val1 = argv[1];
        string val2 = argv[2];
        string val3 = argv[3];
        stringstream vala(val1); stringstream valb(val2); stringstream valc(val3);
        vala>>L_lattice; valb>>M_lattice; valc>>N_lattice;
    }
    PetscInt       n;
    PetscScalar    alpha, beta, AngToBohr=1.88973;
    PetscLogDouble t0 = 0.0, t1 = 0.0;
    //PetscInt       MPI_RANK;
    
    PetscInitialize(&argc, &argv, (char*)0, help);
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
    
    Natoms = L_lattice*M_lattice*N_lattice*9;
    Nelectrons = (L_lattice*M_lattice*N_lattice)*(3*4 + 6*6); // 3 Si atoms : 3*4 || 6 O atoms: 6*6
    nSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);
    if (MPI_RANK==0){
        ierr = PetscTime(&t0); CHKERRQ(ierr);
        cout<<"starting to write coordinates"<<endl;
    }
    arrangeCoordinates();
    if (MPI_RANK==0){
        ierr = PetscTime(&t1); CHKERRQ(ierr);
        cout<<"time arranging Coordinates : "<<(t1-t0)<<endl;
    }
    localHSize = localEndIndex[ni-1] - localBeginIndex[0]; // this size determines the local sizes of vectors and matrices:
    /*cout<<localHSize<<endl;
    for (int i=0;i<ni;i++){
        cout<<"localEndIndex[i] = "<<localEndIndex[i]<<"localBeginIndex[i] "<<localBeginIndex[i]<<endl;
    }*/
    if (MPI_RANK==0){
        ierr = PetscTime(&t0); CHKERRQ(ierr);
        cout<<"starting to constructing hamiltonian "<<localHSize<<endl;
    }
    initializeForH();
    constructH();
    initializeDiagP();
    if (MPI_RANK==0){
        ierr = PetscTime(&t1); CHKERRQ(ierr);
        cout<<"time Hamiltonian : "<<(t1-t0)<<endl;
    }
    
    if (MPI_RANK==0){
        ierr = PetscTime(&t0); CHKERRQ(ierr);
        cout<<"starting to run KPM"<<endl;
    }
    runKPM();
    if (MPI_RANK==0){
        ierr = PetscTime(&t1); CHKERRQ(ierr);
        cout<<"time KPM : "<<(t1-t0)<<endl;
    }
    
    if (MPI_RANK==0){
        ierr = PetscTime(&t0); CHKERRQ(ierr);
        cout<<"starting to run SP2"<<endl;
    }
    computeDensityMatrix();
    if (MPI_RANK==0){
        ierr = PetscTime(&t1); CHKERRQ(ierr);
        cout<<"time SP2 : "<<(t1-t0)<<endl;
    }
    
    
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;


}

