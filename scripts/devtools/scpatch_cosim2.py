# Reliable full-cosim: does ALL work AT df.build interception (like the working
# csynth sweep) -> no dependence on the test calling the returned mod. Self-generates
# inputs from the region signature, runs systemc CSIM (golden) + RTL cosim, diffs,
# then aborts the test. Prints COSIM_RESULT immediately. Cleans /scratch after each.
import os, glob, shutil, tempfile, subprocess, numpy as np
MGC=os.environ.get("MGC_HOME",""); NC="/opt/cadence/XCELIUM2403"; WORK="/scratch/cosim_work"
_busy=[False]; _cur=[None]; _seen=set()
class _Done(Exception): pass
import allo.dataflow as df
_df=df.build
def _rec(k,st,d=""): print(f"COSIM_RESULT | {st:16} | {k.split('::')[-1]:42} | {d}", flush=True)
_DT={"i8":np.int8,"i16":np.int16,"i32":np.int32,"i64":np.int64,"ui8":np.uint8,
     "ui16":np.uint16,"ui32":np.uint32,"ui64":np.uint64,"f16":np.float32,"f32":np.float32,"f64":np.float64}
def _mkarg(ty):
    shape=tuple(getattr(ty,"shape",()) or ()); ds=str(getattr(ty,"dtype","i32"))
    dt=_DT.get(ds,np.int32)
    if dt in (np.float32,np.float64):
        return (np.random.rand(*shape).astype(dt)) if shape else dt(1.0)
    return (np.random.randint(0,8,size=shape).astype(dt)) if shape else dt(1)
def _cosim(build,top,key):
    # `build(mode, project) -> module`. Two call paths reach here: df.build(region,...)
    # and Schedule.build(self,...). The second one matters because a design that applies
    # PRIMITIVES must use it -- hooking only df.build silently skipped every scheduled
    # design (test_systolic, test_systolic_conv, test_mlp, test_unified_systolic).
    pc=tempfile.mkdtemp(prefix="cs_",dir=WORK); ps=tempfile.mkdtemp(prefix="sy_",dir=WORK)
    try:
        try: m=build("csim",pc)
        except Exception as e: _rec(key,"CSIM_BUILD_FAIL",str(e).splitlines()[0][:50]); return
        try: m(*[_mkarg(t) for t in top.__annotations__.values()])
        except Exception as e: _rec(key,"CSIM_RUN_FAIL",str(e).splitlines()[0][:50]); return
        gout=sorted(glob.glob(pc+"/output*.data")); gin=sorted(glob.glob(pc+"/input*.data"))
        if not gout: _rec(key,"NO_GOLDEN",""); return
        try: build("csyn",ps)
        except Exception as e: _rec(key,"CSYN_FAIL",str(e).splitlines()[0][:50]); return
        rt=ps+"/run.tcl"
        if not os.path.exists(rt): _rec(key,"NO_TCL",""); return
        t=open(rt).read()
        t=t.replace("solution options set /Input/CompilerFlags {{-D_GLIBCXX_USE_CXX11_ABI=0}}\n","")
        ins="flow package require /SCVerify\nflow package option set /SCVerify/USE_NCSIM true\nflow package option set /SCVerify/USE_MSIM false\nflow package option set /SCVerify/USE_VCS false\n"
        open(rt,"w").write(t.replace('solution file add',ins+'solution file add',1))
        b=ps+"/cosb"; os.makedirs(b,exist_ok=True)
        try: subprocess.run([f"{MGC}/bin/catapult","-shell","-file",rt],cwd=b,capture_output=True,text=True,timeout=500)
        except subprocess.TimeoutExpired: _rec(key,"SYNTH_TIMEOUT",""); return
        for sh in glob.glob(b+"/**/sysc_sim.h",recursive=True):
            s=open(sh).read()
            if "ac_int.h" not in s: open(sh,"w").write(s.replace("#include <systemc.h>","#include <systemc.h>\n#include <ac_int.h>"))
        mk=glob.glob(b+"/**/Verify_concat_sim_rtl_v_ncsim.mk",recursive=True)
        if not mk: _rec(key,"SYNTH_FAIL","no cosim mk"); return
        v1=os.path.dirname(os.path.dirname(mk[0]))
        for f in gin: shutil.copy(f,b+"/"+os.path.basename(f))
        env=dict(os.environ); env["NC_ROOT"]=NC; env["NCSim_NC_ROOT"]=NC
        try: subprocess.run([f"{MGC}/bin/make","-f","./scverify/Verify_concat_sim_rtl_v_ncsim.mk",f"NC_ROOT={NC}",f"NCSim_NC_ROOT={NC}","SIMTOOL=ncsim","sim"],cwd=v1,capture_output=True,text=True,timeout=600,env=env)
        except subprocess.TimeoutExpired: _rec(key,"COSIM_TIMEOUT",""); return
        rout=sorted(glob.glob(b+"/output*.data"))
        if not rout: _rec(key,"COSIM_NO_OUTPUT",""); return
        # zip() truncates to the shorter list, so a design whose RTL emitted FEWER
        # arrays than the golden would compare only the overlap and report MATCH.
        if len(gout)!=len(rout): _rec(key,"COSIM_ARITY",f"golden {len(gout)} vs rtl {len(rout)}"); return
        try: match=all(open(g).read().split()==open(r).read().split() for g,r in zip(gout,rout))
        except Exception: match=False
        _rec(key,"COSIM_MATCH" if match else "COSIM_MISMATCH",f"{len(gout)} arr")
    finally:
        shutil.rmtree(pc,ignore_errors=True); shutil.rmtree(ps,ignore_errors=True)
def pytest_runtest_setup(item): _cur[0]=item.nodeid

from allo.customize import Schedule
_sb = Schedule.build
_cz = df.customize

def _run(build, top, key):
    _seen.add(key); _busy[0]=True
    try: _cosim(build, top, key)
    except Exception as e: _rec(key,"HARNESS_ERR",str(e)[:50])
    finally: _busy[0]=False
    raise _Done()

def pytest_configure(config):
    def patched_customize(func,*a,**k):
        sch=_cz(func,*a,**k)
        # Schedule keeps no reference to its region, and we need the region's annotations
        # to synthesize arguments. Stash it on the way past.
        try: sch._allo_region=func
        except Exception: pass
        return sch

    def patched(top,*a,**k):
        key=_cur[0] or "?"
        if _busy[0] or key in _seen: return _df(top,*a,**k)
        _run(lambda mode,prj: _df(top,target="systemc",mode=mode,project=prj), top, key)

    def patched_sbuild(self,*a,**k):
        key=_cur[0] or "?"
        top=getattr(self,"_allo_region",None)
        # No region captured (a plain allo.customize kernel, not a df.region) -> we cannot
        # synthesize arguments, so run the build untouched rather than guess.
        if _busy[0] or key in _seen or top is None: return _sb(self,*a,**k)
        _run(lambda mode,prj: _sb(self,target="systemc",mode=mode,project=prj), top, key)

    df.customize=patched_customize
    df.build=patched
    Schedule.build=patched_sbuild
