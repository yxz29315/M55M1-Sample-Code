/**************************************************************************//**
 * @file     MouthLabels.cpp
 * @brief    Labels for mouth detection (mouth closed, mouth open)
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#if defined(__LOAD_MODEL_FROM_SD__)

#include "BufAttributes.hpp"
#include <vector>
#include <string>

static const char *labelsVec[] LABELS_ATTRIBUTE =
{
    "mouth closed",
    "mouth open",
};

bool GetLabelsVector(std::vector<std::string> &labels)
{
    constexpr size_t labelsSz = 2;
    labels.clear();

    if (!labelsSz)
    {
        return false;
    }

    labels.reserve(labelsSz);

    for (size_t i = 0; i < labelsSz; ++i)
    {
        labels.emplace_back(labelsVec[i]);
    }

    return true;
}

#endif /* __LOAD_MODEL_FROM_SD__ */
