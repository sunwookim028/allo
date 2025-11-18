#include "allo/Translation/EmitVivadoHLS.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"

extern "C" void registerAlloVivadoHLSTranslation() {
  static bool initialized = false;
  if (initialized)
    return;
  initialized = true;
  allo::registerEmitVivadoHLSTranslation();
}

__attribute__((constructor)) static void registerPlugin() {
  registerAlloVivadoHLSTranslation();
}




