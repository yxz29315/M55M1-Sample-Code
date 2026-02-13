#include "BufAttributes.hpp"

#include <vector>
#include <string>

static const char *labelsVec[] LABELS_ATTRIBUTE =
{
    "face"
};

bool GetLabelsVector(std::vector<std::string> &labels)
{
    labels.clear();
    labels.reserve(1);
    labels.emplace_back("face");
    return true;
}