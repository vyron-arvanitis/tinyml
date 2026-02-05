#include "matrix.h"
namespace tinyml
{
    class SGD
    {
    public:
        SGD(float lr) : lr_(lr) {}
        void step();

    private:
        float lr_;
    };
}