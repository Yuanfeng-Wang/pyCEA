# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class CEA2:
    def __init__(self):
        self.Rr = 8314.51
        self.dr_avg = 6.0221367
        self.Boltz = 1.380658

        # 原子記号リスト
        self.symbols = ['H','D','HE','LI','BE','B','C','N','O','F','NE','NA',
            'MG','AL','SI','P','S','CL','AR','K','CA','SC',
            'TI','V','CR','MN','FE','CO','NI','CU','ZN','GA','GE','AS',
            'SE','BR','KR','RB','SR','Y','ZR','NB','MO','TC','RU','RH',
            'PD','AG','CD','IN','SN','SB','TE','I','XE','CS','BA','LA',
            'CE','PR','ND','PM','SM','EU','GD','TB','DY','HO','ER','TM',
            'YB','LU','HF','TA','W','RE','OS','IR','PT','AU','HG','TL',
            'PB','BI','PO','AT','RN','FR','RA','AC','TH','PA','U','NP',
            'PU','AM','CM','BK','CF','ES']
        
        # 原子質量リスト
        self.atom_wt = [1.00794, 2.014102, 4.002602, 6.941, 9.012182, 10.811,
            12.0107, 14.0067, 15.9994, 18.9984032, 20.1797, 22.989770,
            24.305, 26.981538, 28.0855, 30.973761, 32.065, 35.453, 39.948,
            39.0983, 40.078, 44.95591, 47.867, 50.9415, 51.9961, 54.938049,
            55.845, 58.933200, 58.6934, 63.546, 65.39, 69.723, 72.64,
            74.92160, 78.96, 79.904, 83.80, 85.4678, 87.62, 88.90585,
            91.224, 92.90638, 95.94, 97.9072, 101.07, 102.9055, 106.42,
            107.8682, 112.411, 114.818, 118.710, 121.760, 127.6, 126.90447,
            131.293, 132.90545, 137.327, 138.9055, 140.116, 140.90765,
            144.9127, 145.0, 150.36, 151.964, 157.25, 158.92534, 162.50,
            164.93032, 167.259, 168.93421, 173.04, 174.967, 178.49, 180.9479,
            183.84, 186.207, 190.23, 192.217, 195.078, 196.96655, 200.59,
            204.3833, 207.2, 208.98038, 208.9824, 209.9871, 222.0176, 223.0197,
            226.0254, 227.0278, 232.0381, 231.03588, 238.02891, 237.0482,
            244.0642, 243.0614, 247.0703, 247.0703, 251.0587, 252.083]

        # 原子バランスリスト
        self.valance = [1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, -2.0, -1.0, 0.0, 1.0,
            2.0, 3.0, 4.0, 5.0, 4.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 3.0,
            2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 3.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            4.0, 5.0, 6.0, 7.0, 4.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, -1.0, 0.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0]

        REAL*8 Enn,Ennl,Enlsav,Ensave,Sumn
      REAL*8 Deln(MAXNGC),Enln(MAXNGC),Sln(MAXNGC)
      REAL*8 En(MAXNGC,NCOL)
      COMMON /COMP  / Deln,En,Enln,Enn,Ennl,Enlsav,Ensave,Sln,Sumn
C
      INTEGER Ip,Iplt,It,Nc,Ng,Ngp1,Nlm,Nplt,Nof,Nomit,Nonly,Np,Npr,Npt,
     &        Ngc,Nsert,Nspr,Nspx,Nt
      INTEGER Jcond(45),Jx(MAXEL),Nfla(MAXR),Ifz(MAXNC)
      COMMON /INDX  / Ip,Iplt,It,Jcond,Jx,Nc,Ng,Ngp1,Nlm,Nplt,Nof,Nomit,
     &                Nonly,Np,Npr,Npt,Ngc,Nsert,Nspr,Nspx,Nt,Nfla,Ifz
C
      REAL*8 Cpmix,Wmix,Bcheck
      REAL*8 Am(2),Hpp(2),Vmin(2),Vpls(2),Wp(2),Atmwt(100),Oxf(MAXMIX),
     &       P(MAXPV),Rh(2),T(MAXT),V(MAXPV),Valnce(100)
      REAL*8 B0p(MAXEL,2)
      COMMON /INPT  / Am,B0p,Cpmix,Hpp,Vmin,Vpls,Wmix,Wp,Atmwt,Bcheck,
     &                Oxf,P,Rh,T,V,Valnce
C
      INTEGER Imat,Iq1,Isv,Jliq,Jsol,Lsave,Msing
      COMMON /MISCI / Imat,Iq1,Isv,Jliq,Jsol,Lsave,Msing
C
      LOGICAL Convg,Debug(NCOL),Detdbg,Detn,Eql,Gonly,Hp,Ions,Massf,
     &        Moles,Newr,Pderiv,Shock,Short,Siunit,Sp,Tp,Trnspt,Vol
      COMMON /MISCL / Convg,Debug,Detdbg,Detn,Eql,Gonly,Hp,Ions,Massf,
     &                Moles,Newr,Pderiv,Shock,Short,Siunit,Sp,Tp,Trnspt,
     &                Vol
C
      REAL*8 Avgdr,Boltz,Eqrat,Hsub0,Oxfl,Pi,Pp,R,Rr,Size,S0,Tln,Tm,
     &       Trace,Tt,Viscns,Vv
      REAL*8 Atwt(MAXEL),B0(MAXEL),X(MAXMAT)
      REAL*8 A(MAXEL,MAXNGC),G(MAXMAT,MAXMAT+1)
      COMMON /MISCR / A,Atwt,Avgdr,Boltz,B0,Eqrat,G,Hsub0,Oxfl,Pi,Pp,R,
     &                Rr,Size,S0,Tln,Tm,Trace,Tt,Viscns,Vv,X
C
      CHARACTER*2 Elmt(MAXEL),Ratom(MAXR,12),Symbol(100)
      CHARACTER*4 Fmt(30)
      CHARACTER*8 Fox(MAXR)
      CHARACTER*10 Thdate
      CHARACTER*15 Case,Energy(MAXR),Omit(0:MAXNGC),Pltvar(20),
     &             Prod(0:MAXNGC),Rname(MAXR)
      CHARACTER*200 Pfile
      COMMON /CDATA / Case,Elmt,Energy,Fmt,Fox,Omit,Pltvar,Prod,Ratom,
     &                Rname,Symbol,Thdate,Pfile
C
      REAL*8 Cpr(NCOL),Dlvpt(NCOL),Dlvtp(NCOL),Gammas(NCOL),Hsum(NCOL),
     &       Ppp(NCOL),Ssum(NCOL),Totn(NCOL),Ttt(NCOL),Vlm(NCOL),
     &       Wm(NCOL)
      REAL*8 Pltout(500,20)
      COMMON /PRTOUT/ Cpr,Dlvpt,Dlvtp,Gammas,Hsum,Ppp,Ssum,Totn,Ttt,Vlm,
     &                Wm,Pltout
C
      INTEGER Nreac
      INTEGER Jray(MAXR)
      REAL*8 Dens(MAXR),Enth(MAXR),Pecwt(MAXR),Rmw(MAXR),Rtemp(MAXR)
      REAL*8 Rnum(MAXR,12)
      COMMON /REACTN/ Dens,Enth,Pecwt,Rmw,Rnum,Rtemp,Jray,Nreac
C
      REAL*8 Cpsum
      REAL*8 Cft(MAXNC,9),Coef(MAXNG,9,3),Temp(2,MAXNC)
      REAL*8 Cp(MAXNGC),H0(MAXNGC),Mu(MAXNGC),Mw(MAXNGC),S(MAXNGC),Tg(4)
      COMMON /THERM / Cft,Coef,Cp,Cpsum,H0,Mu,Mw,S,Temp,Tg
C
      INTEGER Iopt,Isup,Nfz,Npp,Nsub,Nsup
      LOGICAL Area,Debugf,Fac,Froz,Page1,Rkt
      REAL*8 Acat,Awt,Cstr,Tcest,Ma
      REAL*8 Aeat(NCOL),App(NCOL),Pcp(2*NCOL),Sonvel(NCOL),Spim(NCOL),
     &       Subar(13),Supar(13),Vmoc(NCOL)
      COMMON /ROCKT / Acat,Aeat,App,Awt,Cstr,Ma,Pcp,Sonvel,Spim,Subar,
     &                Supar,Vmoc,Tcest,Area,Iopt,Debugf,Fac,Froz,Isup,
     &                Nfz,Npp,Nsub,Nsup,Page1,Rkt
C
      INTEGER Nsk
      LOGICAL Incdeq,Incdfz,Refleq,Reflfz,Shkdbg
      REAL*8 U1(NCOL),Mach1(NCOL),A1,Gamma1
      COMMON /SHOCKS/ U1,Mach1,A1,Gamma1,Incdeq,Incdfz,Refleq,Reflfz,
     &                Shkdbg,Nsk
C
      INTEGER Nm,Nr,Ntape
      INTEGER Ind(MAXTR),Jcm(MAXEL)
      REAL*8 Cprr(MAXTR),Con(MAXTR),Wmol(MAXTR),Xs(MAXTR)
      REAL*8 Eta(MAXTR,MAXTR),Stc(MAXTR,MAXTR)
      COMMON /TRNP  / Cprr,Con,Eta,Wmol,Xs,Stc,Ind,Jcm,Nm,Nr,Ntape
C
      REAL*8 Coneql(NCOL),Confro(NCOL),Cpeql(NCOL),Cpfro(NCOL),
     &       Preql(NCOL),Prfro(NCOL),Vis(NCOL)
      COMMON /TRPTS / Coneql,Confro,Cpeql,Cpfro,Preql,Prfro,Vis



    def frozen(self):
        # L1666
        pass

    def output1(self):
        # L3009
        pass

    def rkt_output(self):
        # L3682
        pass

    def rocket(self):
        # L3906
        self.acatsv = 0
        self.aeatl = 0
        self.appl = 0
        self.aratio = 0
        self.asq = 0
        self.check = 0
        self.cprf = 0
        self.dd = 0
        self.dh = 0
        self.dlnp = 0
        self.dlnpe = 0
        self.dlt = 0
        self.done = 0
        self.dp = 0
        self.eln = 0
        self.i = 0
        self.i01 = 0
        self.i12 = 0
        self.iof = 0
        self.iplt1 = 0
        self.iplte = 0
        self.ipp = 0
        self.isub = 0
        self.isup1 = 0
        self.isupsv = 0
        self.itnum = 0
        self.itrot = 0
        self.mat = 0
        self.msq = 0
        self.nar = 0
        self.nipp = 0
        self.niter = 0
        self.nn = 0
        self.npr1 = 0
        self.nptth = 0
        self.p1 = 0
        self.pcpa = 0
        self.pcplt = 0
        self.pinf = 0
        self.pinj = 0
        self.pinjas = 0
        self.pjrat = 0
        self.ppa = 0
        self.pr = 0
        self.pracat = 0
        self.prat = 0
        self.pratsv = 0
        self.pvg = 0
        self.seql = 0
        self.test = 0
        self.thi = 0
        self.tmelt = 0
        self.usq = 0

        a1l = -1.26505
        b1 = 1.0257
        c1 = -1.2318
        pa = 1.0e5


    