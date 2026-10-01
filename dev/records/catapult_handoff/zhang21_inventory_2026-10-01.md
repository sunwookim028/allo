# Catapult SystemC toolchain on zhang-21 — inventory, 2026-10-01

Read-only version queries by the `zhang21` agent session. Nothing was changed on
the host. These are the versions `scripts/systemc-csim-setup.sh` pins its
stand-in to.

## Catapult
- `MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home`. An older install is also
  present: `/opt/siemens/catapult/2024.1_2-1117371/Mgc_home`.
  `module load catapult-2024` sets `MGC_HOME`, `MGLS_LICENSE_FILE`,
  `SALT_LICENSE_SERVER` and `CDS_LIC_FILE`.
- `catapult -version`: Catapult Ultra Synthesis 2024.2/1130128 (Production
  Release), Mon Aug 26 21:59:12 PDT 2024.
- Licence: with `MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu`,
  `catapult -shell -file` printed LIC-13 and then LIC-14 (checked out) on
  2026-10-01.

## Bundled under `$MGC_HOME/shared`
| library | version | hlslibs/Accellera tag |
| --- | --- | --- |
| SystemC | Accellera 2.3.3 (`SYSTEMC_VERSION 20181013`). TLM 2.0.5, SCV 2.0.0. `shared/lib/libsystemc-2.3.3.so`, with per-compiler builds in `shared/lib/Linux/{gcc-10.3.0-64,gcc-4.9.2-64,clang-13.0.0-64}` | 2.3.3 |
| MatchLib Connections | `connections/connections.h`, revision history up to "2.2.0 - CAT-34924 ... CAT-37259". `CONNECTIONS_ACCURATE_SIM` is the default | 2.2.0 |
| ac_types | Software Version 4.9, Release Build 4.9.0 (Aug 25 2024). `ac_int.h` still defines `AC_VERSION 4` / `AC_VERSION_MINOR 8` | 4.9.0 |
| ac_simutils | `mc_scverify.h`: Software Version 1.6, Build 1.6.0 (Feb 21 2024) | 1.6.0 |
| ac_math | 3.6 | — |
| g++ | `$MGC_HOME/bin/g++`: "g++ (Calypto) 10.3.0"; lib dir `shared/lib/Linux/gcc-10.3.0-64` | — |

## Simulators
- Xcelium: `xrun` 24.03-s005 at `/opt/cadence/XCELIUM2403/tools.lnx86/bin/xrun`,
  with `CDS_LIC_FILE=5280@en-license-05.coecis.cornell.edu` and `LD_PRELOAD`
  unset. A trivial module ran to `$finish` on 2026-10-01.
  Gotcha: `module load catapult-2024` appends a non-existent
  `/opt/cadence/XCELIUM2409/...`, so add the 2403 `bin` to `PATH` by hand.
- SCVerify: flows `app_ncsim.flo` (Xcelium), `app_osci.flo` and
  `app_questasim.flo` are present. Questa is not usable
  (`/opt/siemens/Questa/2024.2` holds only QVIP). SCVerify with Xcelium last ran
  on 2026-09-24, for the C++ design `ppa_mac16`, not a SystemC design.

## Host
- RHEL 8.10, glibc 2.28, system gcc 8.5.0.
- One allo checkout: `/work/shared/users/phd/sk3463/allo`, on `main` at
  `d7377398` (equal to `origin/main` at the time).
