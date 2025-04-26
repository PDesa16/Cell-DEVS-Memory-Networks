#ifndef GRID_CELL_FACTORY_BASE_HPP
#define GRID_CELL_FACTORY_BASE_HPP

#include <cadmium/celldevs/grid/cell.hpp>
#include <cadmium/celldevs/grid/config.hpp>

using namespace cadmium::celldevs;

template<typename StateT>
class GridCellFactoryBase {
public:
    virtual ~GridCellFactoryBase() = default;

    virtual std::shared_ptr<GridCell<StateT, double>> create(
        const coordinates& id,
        const std::shared_ptr<const GridCellConfig<StateT, double>>& config
    ) const = 0;
};


#endif
