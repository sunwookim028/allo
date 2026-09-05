/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The naming vocabulary: the one place a hardware identifier is composed. Every
// module, boundary port, storage cell and instance name is built here, from one
// grammar:
//
//   name      := <owner> ( "_" <qualifier> )* [ "_" <field> ]
//   owner     := the source identifier (a NameLoc), else a structural token:
//                a<argNo> / m<memId> / u<unitId> / ch<chanId>
//   qualifier := <letters><number>, the number bound with no separator
//                (rd0, wr1, st, b3)
//   field     := addr | data | we | valid | ready | in | out
//
// Two rules keep the names stable, which the port manifest (the C++/Python
// contract) needs:
//
//   1. A group index is emitted unconditionally wherever the EMITTER decides
//      the count, i.e. a memory argument's port groups, and never where the
//      source signature fixes it (a scalar, a stream, a result).
//   2. A fallback keys on the owner's own id, never on a position in the port
//      list, so adding a port to one argument cannot rename another's.
//
// `verilogName` escapes anything ExportVerilog would rewrite, so the manifest,
// authored before LegalizeNames runs, equals the emitted Verilog. Emitters call
// these functions and never concatenate a name themselves.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_NAMING_H
#define ALLO_MICROARCH_NAMING_H

#include "allo/Microarch/Datapath.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// The fixed control ABI. Every emitted module carries exactly these four ports,
// spelled once here and published in the port manifest.
//===----------------------------------------------------------------------===//
constexpr llvm::StringLiteral kClk = "clk";
constexpr llvm::StringLiteral kRst = "rst";
constexpr llvm::StringLiteral kStart = "start";
constexpr llvm::StringLiteral kDone = "done";
/// An extern operator module's own fixed ports: data inputs `a`, `b`, ..., then
/// `clk`, then `ce` under a clock-enabled stall contract, then the result.
constexpr llvm::StringLiteral kCe = "ce";
constexpr llvm::StringLiteral kOpOut = "y";

/// A source-derived string made safe as a *final* SystemVerilog identifier.
/// Illegal characters become '_', and a name ExportVerilog would rewrite (a
/// keyword such as `input`, `wire`, `buf`) gets a trailing '_'. Applied to the
/// composed name, so an `output` array yields `output_wr0`, not `output__wr0`.
std::string verilogName(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// Owner tokens: the one structural-fallback vocabulary.
//===----------------------------------------------------------------------===//

/// The owner token of a boundary value: its source name (NameLoc), else
/// `a<argNo>` when it is a kernel argument, else \p fallback. Charset-sanitized
/// but not keyword-escaped, since the escape belongs to the composed name.
std::string ownerOf(Value v, llvm::StringRef fallback);
std::string ownerOf(Location loc, llvm::StringRef fallback);
/// `ownerOf` disambiguated against \p siblings, the values whose owner tokens
/// share a namespace (every memref of a module, every scalar argument). A value
/// tying with a sibling takes a suffix unique by construction: its argument
/// position for an argument, \p fallback otherwise.
std::string uniqueOwnerOf(Value v, llvm::ArrayRef<Value> siblings,
                          llvm::StringRef fallback);
std::string argOwner(unsigned argNo); // a2
std::string memOwner(MemId m);        // m0
std::string unitOwner(UnitId u);      // u5
std::string chanOwner(StreamId s);    // ch1
std::string regOwner(RegId r);        // reg3
std::string regionTagOf(unsigned r);  // r3, the prefix of a region's own cells

//===----------------------------------------------------------------------===//
// Field suffixes: the leaves of every port name.
//===----------------------------------------------------------------------===//

std::string portAddr(llvm::StringRef base);
std::string portData(llvm::StringRef base);
std::string portWe(llvm::StringRef base);
std::string portValid(llvm::StringRef base);
std::string portReady(llvm::StringRef base);

//===----------------------------------------------------------------------===//
// Port-group bases. The primitives compose a base from (owner, role, index);
// the resolvers derive those from the datapath.
//===----------------------------------------------------------------------===//

/// `<owner>_rd<i>` / `<owner>_wr<i>`: one memory access's port group.
std::string memBase(llvm::StringRef owner, bool write, unsigned group);
/// `<owner>_st`: a stream channel's handshake group.
std::string streamBase(llvm::StringRef owner);
/// `<owner>_in`: a scalar argument port (a whole port, no field).
std::string scalarBase(llvm::StringRef owner);
/// `<owner>_out`: a scalar result port (a whole port, no field).
std::string resultBase(llvm::StringRef owner);
/// `<base>_b<k>`: one bank of a partitioned array, port group or storage cell.
std::string bankBase(llvm::StringRef base, unsigned bank);
/// Which direction of a scattered element port a name is for. `Only` takes the
/// bare name, for an argument the kernel just reads or just writes; an argument
/// used BOTH ways needs its two ports told apart.
enum class ElemDir { Only, In, Out };

/// `<owner>_<k>`, plus `_in` / `_out` for a read-write argument: element \p k
/// of a scattered argument (`MemUnit::scattered`). A whole port on its own for
/// a read; the group base a write's `_we` hangs off for a write. The index is
/// BARE, unlike every other qualifier here, because the argument's type fixes
/// the count.
std::string elemBase(llvm::StringRef owner, unsigned index,
                     ElemDir dir = ElemDir::Only);

/// The boundary interfaces of one external access, as (bank, base): one entry
/// for an unbanked or statically-routed access, one per bank for a
/// data-dependent one. The base is `acc.portBase`, composed once by
/// `enumerateBoundaryPorts`; this only expands it across banks.
llvm::SmallVector<std::pair<unsigned, std::string>>
extPorts(const MemUnit &m, const MemUnit::Access &acc);
/// A stream channel's port base. Two stream arguments of one module can share a
/// source name (a systolic PE gets `fifo[i,j]` and puts `fifo[i,j+1]`); a
/// colliding group splits by direction, then by channel id.
std::string streamPortBase(const Datapath &dp, const StreamChannel &s);
/// A scalar argument's port.
std::string scalarPortName(const Datapath &dp, const IOPort &io);
/// Result \p i of \p n: `ret_out`, or `ret<i>_out` for a multi-result kernel.
std::string resultPortName(unsigned i, unsigned n);

//===----------------------------------------------------------------------===//
// Internal cells and instances. These names reach waveforms and the netlist but
// never the manifest, so CIRCT is free to uniquify them.
//===----------------------------------------------------------------------===//

/// What an array is called before any bank, instance or field suffix: a
/// MemUnit's owner token, disambiguated against the module's other memrefs.
std::string memOwnerName(const Datapath &dp, const MemUnit &m);
/// On-chip storage for the buffer named \p owner: bank \p bank when it is one
/// of \p numBanks, and instance \p inst when the bank is held in \p instances
/// of them. An index that has only one value is left off. The Datapath overload
/// resolves a MemUnit's owner name and instance count first.
std::string memCellName(llvm::StringRef owner, unsigned numBanks, unsigned bank,
                        unsigned instances = 1, unsigned inst = 0);
std::string memCellName(const Datapath &dp, const MemUnit &m, unsigned bank,
                        unsigned inst = 0);
/// The buffer's own name, carrying no instance index: what the report calls the
/// array and what a reader looks it up by.
std::string memArrayName(const Datapath &dp, const MemUnit &m);
/// `<owner>_<k>`: element \p k of an internal array scattered into registers.
std::string memElemName(const Datapath &dp, const MemUnit &m, unsigned k);
/// `r<region>_<sig>`: a region's control-plane signal (`run`, `issue`, `iv`,
/// `phase`, `done`, `ce`). The StringRef form takes an already-formed tag
/// (`EmitContext::regionTag`).
std::string regionSignal(unsigned region, llvm::StringRef sig);
std::string regionSignal(llvm::StringRef tag, llvm::StringRef sig);
/// `<owner>_d<k>`: tap \p k of a delay chain, i.e. \p owner delayed k cycles.
std::string regTapName(llvm::StringRef owner, unsigned k);
/// `r<region>_sv<k>`: a region's survivor or loop-carried iter-arg latch.
std::string survivorName(unsigned region, unsigned k);
/// `<owner>_u<id>`, else `u<id>`: a compute-unit instance. ExportVerilog names
/// an instance's results `_<instance>_<port>` and ignores a namehint on the
/// instance, so the source name has to be folded in here.
std::string unitInstanceName(const FuncUnit &u);
/// `<callee>_i<n>`: a child-kernel instance, indexed so two invocations of one
/// callee stay distinct.
std::string childInstanceName(llvm::StringRef callee, unsigned n);
/// `<chan>_<sig>`: a composed channel's own signal. Covers only the shim built
/// here; a `seq.fifo`'s own internals are named by CIRCT's lowering.
std::string channelSignal(llvm::StringRef chan, llvm::StringRef sig);

/// The extern operator-module name for an IP-realized unit: the `dcp.operator`
/// symbol its `OperatorIdentity` names, a floating-point compare additionally
/// encoding its predicate. Not passed through `verilogName`: it must stay the
/// device's string for the simulation model to join on it.
std::string operatorModuleName(const FuncUnit &u);
/// The predicate an operator module name encodes (a floating-point compare's
/// `ogt`), empty otherwise. Published in the manifest so the simulation model
/// joins on it.
std::string operatorPredicate(const FuncUnit &u);

/// Attach a readable Verilog name to \p v, derived from \p loc's NameLoc, so a
/// frontend variable keeps its source name instead of CIRCT's `_GEN` fallback.
/// A no-op when \p loc carries no name or when \p v is a block argument, which
/// the port interface names instead.
void nameValue(Value v, Location loc);
/// Attach \p name directly, for a name held as a string rather than on a
/// Location. A no-op if empty or if \p v is not an op result.
void nameValue(Value v, llvm::StringRef name);

/// Whether \p v already carries a name through either channel `nameValue`
/// writes. One wire can serve two control roles, and the first name assigned
/// is the one kept.
bool isNamedValue(Value v);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_NAMING_H
