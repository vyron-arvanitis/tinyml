#include "matrix.h"
namespace tinyml
{   


    class Optimizer{

        public:
            Optimizer();
            virtual ~Optimizer() = default;

            virtual void step()=0;


    };

    class SGD final :  public Optimizer
    {
    public:
        void step() override;

    private:
        float lr_;
    };
}