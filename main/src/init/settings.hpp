/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Translation unit for the simulation initializer library
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "io/ifile_io.hpp"
#include "io/id_tag_utils.hpp"

namespace sphexa
{

using ScalarValue = double;
using VectorValue = std::vector<IdType>; // TODO: update name
using VVectorValue = std::vector<VectorValue>; // TODO: update name
using FVectorValue = std::vector<ScalarValue>;

//! @brief Wrapper for scalar-type and vector-type parameters
struct Param {

    std::variant<ScalarValue, VectorValue, VVectorValue, FVectorValue> value;

public:

    Param(void) : value(ScalarValue{}) {}

    // Conversion of all arithmetic types to ScalarValue
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>>>
    Param(T v) : value(static_cast<ScalarValue>(v)) {}

    Param(const VectorValue& v) : value(v) {}
    Param(VectorValue&& v) : value(std::move(v)) {}

    Param(const VVectorValue& v) : value(v) {}
    Param(VVectorValue&& v) : value(std::move(v)) {}

    Param(const FVectorValue& v) : value(v) {}
    Param(FVectorValue&& v) : value(std::move(v)) {}

    // Conversion of all arithmetic types to ScalarValue
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>>>
    Param& operator=(T v) { value = static_cast<ScalarValue>(v); return *this; }

    Param& operator=(const VectorValue& v) { value = v; return *this; }
    Param& operator=(VectorValue&& v) { value = std::move(v); return *this; }

    Param& operator=(const VVectorValue& v) { value = v; return *this; }
    Param& operator=(VVectorValue&& v) { value = std::move(v); return *this; }

    Param& operator=(const FVectorValue& v) { value = v; return *this; }
    Param& operator=(FVectorValue&& v) { value = std::move(v); return *this; }

    const auto& getValue() const { return value; }
    auto& getValue() { return value; }

    // Check stored type
    bool isScalar() const { return std::holds_alternative<ScalarValue>(value); }
    bool isVector() const { return std::holds_alternative<VectorValue>(value); }
    bool isVVector() const { return std::holds_alternative<VVectorValue>(value); }
    bool isFVector() const { return std::holds_alternative<FVectorValue>(value); }

    // Implicit conversion to std::variant
    operator std::variant<ScalarValue, VectorValue, VVectorValue, FVectorValue>() const { return value; }

};

using InitSettings = std::map<std::string, Param>;

// struct IdSelectionSettings
// {
//     // TODO: does it make sense to have an id-range based selection?
//     using IdSelectionType = std::variant<IdSelectionList, IdSelectionSphere>;

//     // TODO: this data is already stored in settings_,
//     // can we store here the just keys to retrieve the stored data?
//     IdSelectionType selectionData;

//     int selectionTimeStep;
// };

//using IdSubsets = std::map<std::string, IdSelectionSettings>;
// using IdSubsets = std::vector<IdSelectionSettings>;

//! @brief write @p InitSettings as file attributes of a new file @p path
inline void writeSettings(const InitSettings& settings, const std::string& path, IFileWriter* writer)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error("Cannot write settings: file " + path + " already exists\n");
    }

    writer->addStep(0, 0, path, true);
    for (auto it = settings.cbegin(); it != settings.cend(); ++it)
    {
        if(std::holds_alternative<ScalarValue>(it->second.value)) {
            writer->fileAttribute(it->first, &(std::get<ScalarValue>(it->second.value)), 1);
        }
        else {
            // TODO: if the id selection list is removed after init, this line is never called. Otherwise, we need to define what we want
            writer->fileAttribute(it->first, std::get<FVectorValue>(it->second.value).data(), std::get<FVectorValue>(it->second.value).size());
        }
    }
    writer->closeStep();
}

// //! @brief write @p IdSubsets as file attributes of a new file @p path
// inline void writeSettings(const IdSubsets& idSubsets, const std::string& path, IFileWriter* writer)
// {
//     if (std::filesystem::exists(path))
//     {
//         throw std::runtime_error("Cannot write settings: file " + path + " already exists\n");
//     }

//     writer->addStep(0, 0, path, true);
//     for (auto it = idSubsets.cbegin(); it != idSubsets.cend(); ++it)
//     {   
//         if(std::holds_alternative<IdSelectionSphere>(it->second.selectionData)) {
//             const IdSelectionSphere& idSelectionSphere(std::get<IdSelectionSphere>(it->second.selectionData));
//             // writer->fileAttribute(it->first + "_radius", &(idSelectionSphere.radius), 1);
//             // writer->fileAttribute(it->first + "_center_x", &(idSelectionSphere.center[0]), 1);
//             // writer->fileAttribute(it->first + "_center_y", &(idSelectionSphere.center[1]), 1);
//             // writer->fileAttribute(it->first + "_center_z", &(idSelectionSphere.center[2]), 1);
//             writer->fileAttribute("id_selection_spheres_radius", &(idSelectionSphere.radius), 1);
//             writer->fileAttribute("id_selection_spheres_center_x", &(idSelectionSphere.center[0]), 1);
//             writer->fileAttribute("id_selection_spheres_center_y", &(idSelectionSphere.center[1]), 1);
//             writer->fileAttribute("id_selection_spheres_center_z", &(idSelectionSphere.center[2]), 1);
//         }
//         else if(std::holds_alternative<IdSelectionList>(it->second.selectionData)) {
//             const IdSelectionList& idSelectionList(std::get<IdSelectionList>(it->second.selectionData));
// //            writer->fileAttribute(it->first + "_id_subset", idSelectionList.data(), idSelectionList.size());
//             writer->fileAttribute(it->first + "_id_subset", idSelectionList.data(), idSelectionList.size());
//         }
//         else {
//             throw std::runtime_error("Cannot write settings: unsupported id subset type\n");
//         }
//         writer->fileAttribute(it->first + "_time_step", &(it->second.selectionTimeStep), 1);
//     }
//     writer->closeStep();
// }

//! @brief write @p IdSelections as file attributes of a new file @p path
inline void writeSettings(const IdSelections& idSelections, const std::string& path, IFileWriter* writer)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error("Cannot write settings: file " + path + " already exists\n");
    }

    writer->addStep(0, 0, path, true);
    
    // Write selection spheres
    if(idSelections.hasSpheres()) {
        const auto& spheres = idSelections.spheres;
        writer->fileAttribute("id_selection_spheres_center_x", spheres.center_x.data(), spheres.center_x.size());
        writer->fileAttribute("id_selection_spheres_center_y", spheres.center_y.data(), spheres.center_y.size());
        writer->fileAttribute("id_selection_spheres_center_z", spheres.center_z.data(), spheres.center_z.size());
        writer->fileAttribute("id_selection_spheres_radius", spheres.radius.data(), spheres.radius.size());
        writer->fileAttribute("id_selection_spheres_time_step", spheres.step.data(), spheres.step.size());
    }

    // Write selection lists
    if(idSelections.hasLists()) {
        const auto& lists = idSelections.lists;
        writer->fileAttribute("id_selection_lists_id_subset", lists.lists[0].data(), lists.lists[0].size());
        writer->fileAttribute("id_selection_lists_time_step", lists.step.data(), lists.step.size());
    }

    writer->closeStep();
}


//! @brief Used to initialize particle dataset attributes from builtin named test-cases
class BuiltinWriter
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinWriter(InitSettings attrs)
        : attributes_(std::move(attrs))
    {
    }

    [[nodiscard]] static int rank() { return -1; }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) {
            if(std::holds_alternative<ScalarValue>(attributes_.at(key).getValue())) {
                *arg = std::get<ScalarValue>(attributes_.at(key).getValue());
            }
            else {
                // TODO: I don't have implemented this case because it seems
                // not to be used in the current version of the code
                std::runtime_error("BuiltinWriter: unsupported type");
            }
        }, val);
    };

private:
    InitSettings attributes_;
};

} // namespace sphexa
