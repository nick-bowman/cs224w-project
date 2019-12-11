#ifndef CTDNE_H
#define CTDNE_H

#include "stdafx.h"

#include "Snap.h"
#include "biasedrandomwalk.h"
#include "word2vec.h"

void ctdne(PWNet& InNet, const double& ParamP, const double& ParamQ,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  const bool& OutputWalks, TVVec<TInt, int64>& WalksVV,
  TIntFltVH& EmbeddingsHV);

/// Version for weighted graphs. Edges must have TFlt attribute "weight". No walk output flag. For backward compatibility.
void ctdne(const PNEANet& InNet, const double& ParamP, const double& ParamQ,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
 TIntFltVH& EmbeddingsHV);
#endif //CTDNE_H
