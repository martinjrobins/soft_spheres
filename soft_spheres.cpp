
#include <random>

#include "Aboria.h"
using namespace Aboria;

#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkDoubleArray.h>

#include "boost/program_options.hpp" 
#include <boost/math/constants/constants.hpp>
namespace po = boost::program_options;

#include <math.h>
#include <stdexcept>


int main(int argc, char **argv) {
    const double L = 1.0;

    double2 low(-L/2);
    double2 high(L/2);
    const int nx = 101;
    const int nr = 100;
    const int ntheta = 100;
    const double ds = L/nx;
    const double dr = 0.5*L/nr;
    const double PI = boost::math::constants::pi<double>();
    const double dtheta = 2*PI/ntheta;
    const double D = 1.0;
    double epsilon;
    unsigned int nout;
    const double F0 = 5;
    //const double cutoff = -std::log(1e-7);
    std::string output_name;

    unsigned int n = 20;
    unsigned int samples = 1000;
    double cutoff_ratio,timestep_ratio,final_time,sigma;
    int oned,mode,output_point_positions,init;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("particles", po::value<unsigned int>(&n)->default_value(20), "number of particles per sample")
        ("samples", po::value<unsigned int>(&samples)->default_value(100), "number of samples")
        ("nout", po::value<unsigned int>(&nout)->default_value(10), "number of output points")
        ("cutoff", po::value<double>(&cutoff_ratio)->default_value(5), "cutoff for force calculation divided by epsilon")
        ("epsilon", po::value<double>(&epsilon)->default_value(0.01), "epsilon for force calculation")
        ("sigma", po::value<double>(&sigma)->default_value(0.1), "width of initial condition")
        ("dt", po::value<double>(&timestep_ratio)->default_value(0.33), "average diffusion timstep divided by epsilon")
        ("final-time", po::value<double>(&final_time)->default_value(0.05), "total simulation time")
        ("output-point-positions", po::value<int>(&output_point_positions)->default_value(0), "output point positions")
        ("output-name", po::value<std::string>(&output_name)->default_value("output"), "output file basename")
    ;

    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);  

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }


    const double delta = 0.2*epsilon;
    const double cutoff = cutoff_ratio*epsilon;
    const double mean_s = timestep_ratio*epsilon;
    const double dt = std::pow(mean_s,2)/(2*D);
    const double timesteps = final_time/dt;

    std::cout << "simulating for "<<timesteps<<" timesteps"<<std::endl;
    std::cout << "cutoff/epsilon = "<<cutoff/epsilon<<std::endl;
    std::cout << "force at cutoff / max(force) = "<<std::exp(-cutoff/epsilon)/std::exp(0)<<std::endl;


    /*
     * setup data structures and initial conditions
     */
    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    ABORIA_VARIABLE(force,double2,"force")
    std::vector<unsigned int> output_hist[nout+1];

    for (int s=0;s<=nout;s++) {
        output_hist[s].resize(nx*nx,0);
    }

    int n_samples_complete = 0;
    #pragma omp parallel for shared (output_hist,n_samples_complete)
    for (unsigned int sample=0;sample<samples;sample++) {

        generator_type generator(sample);
        typedef Particles<std::tuple<force>,2> spheres_type;
        typedef spheres_type::position position;
        spheres_type spheres(samples + sample*n);

        std::vector<unsigned int> output_hist_sample(nx*nx,0);

        std::normal_distribution<double> distribution(0,sigma);


        for (int i=0; i<n; ++i) {
            bool not_done = true;
            double2 p;
            while (not_done) {
                p = double2(distribution(generator),distribution(generator));
                not_done = ((p[0] < low[0]) || (p[0] >= high[0]) || (p[1] < low[1]) || (p[1] >= high[1]));
            }
            spheres.push_back(p);
        }

        double2 high_twod = double2(high[0]+(high[0]-low[0]),high[1]+(high[1]-low[1]));
        double2 low_twod = double2(low[0]-(high[0]-low[0]),low[1]-(high[1]-low[1]));
        spheres.init_neighbour_search(low_twod,high_twod,cutoff,bool2(false));

        Symbol<position> p;
        Symbol<force> f;
        Symbol<id> id_;
        Symbol<alive> alive_;
        Label<0,spheres_type> a(spheres);
        Label<1,spheres_type> b(spheres);
        auto dx = create_dx(a,b);
        Normal N;
        VectorSymbolic<double,2> vector;
        Accumulate<std::plus<double2> > sum;
        Accumulate<std::plus<int> > sum_int;

        /*
         * Diffusion step and soft sphere potential
         */
        int timesteps_per_out = 0;
        for (int i = 0; i <= nout; i++) {
            
            for (int j = 0; j < timesteps_per_out; ++j) {
                f[a] = -2*vector(p[a]) + sum(b, norm(dx) < cutoff && id_[a]!=id_[b], 
                         if_else(norm(dx)!=0,(1/epsilon)*exp(-norm(dx)/epsilon)*dx/norm(dx),0));
                p[a] += std::sqrt(2*D*dt)*vector(N,N) + dt*f[a]/(1+norm(f[a])*dt);
                p[a] += vector(if_else(p[a][0] > L/2,L-p[a][0],p[a][0]), if_else(p[a][1] > L/2,L-p[a][1],p[a][1]));
                    + vector(if_else(p[a][0] < -L/2,-L-p[a][0],p[a][0]), if_else(p[a][1] < -L/2,-L-p[a][1],p[a][1]));

            }

            timesteps_per_out = timesteps/nout;

            output_hist_sample.assign(nx*nx,0);

            /*
             * output observables (rdf and concentration)
             */
            for (auto a:spheres) {
                const double2 p_a = get<position>(a);
                const int i = std::floor((p_a[0]-low[0])/ds);
                const int j = std::floor((p_a[1]-low[1])/ds);
                const int index = i*nx+j;
                const double2 dx = p_a-(high-low)/2;
                const double r = dx.norm();
                const double theta = std::atan2(dx[1],dx[0]);
                if (index < output_hist_sample.size()) {
                    output_hist_sample[index]++;
                }
            }

            if (output_point_positions) {
                char buffer[100];
                sprintf(buffer,"%s_points",output_name.c_str());
                spheres.copy_to_vtk_grid(grid);
                vtkWriteGrid(buffer,i,grid);

            }


            /*
             * accumulate into output 
             */
            for (int ii=0; ii < output_hist[i].size(); ii++) {
                unsigned int &tmp1 = output_hist[i][ii];
                #pragma omp atomic 
                tmp1 += output_hist_sample[ii];
            }
        }

        #pragma omp atomic 
        n_samples_complete++;

        std::cout << "samples complete: " << double(n_samples_complete)/samples*100<<" percent"<<std::endl;
    }

    /*
     * all finished, output to files...
     */

    double scaleby = 1.0/(samples*n*ds*ds);

    for (int outi=0;outi<=nout;outi++) {
        std::cout << " writing output file "<<outi<<std::endl;
        vtkSmartPointer<vtkStructuredGrid> structuredGrid =
            vtkSmartPointer<vtkStructuredGrid>::New();
     
        vtkSmartPointer<vtkPoints> points =
                  vtkSmartPointer<vtkPoints>::New();

        for (int i=0;i<nx;i++) {
            for (int j=0;j<nx;j++) {
                points->InsertNextPoint(low[0]+(i+0.5)*ds,low[1]+(j+0.5)*ds,0);
            }
        }

        structuredGrid->SetDimensions(nx,nx,1);
        structuredGrid->SetPoints(points);

        vtkSmartPointer<vtkDoubleArray> array = 
            vtkSmartPointer<vtkDoubleArray>::New();
        array->SetNumberOfComponents(1); 
        array->SetNumberOfTuples(structuredGrid->GetNumberOfPoints());
        for (int i=0;i<output_hist[outi].size();i++) {
            array->SetValue(i,output_hist[outi][i]*scaleby);
        }
        array->SetName("concentration");
        structuredGrid->GetPointData()->AddArray(array);


        // Write file
        vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
        vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
        char buffer[100];
        sprintf(buffer,"%s_hist_%05d.vts",output_name.c_str(),outi);
        writer->SetFileName(buffer);
#if VTK_MAJOR_VERSION <= 5
        writer->SetInput(structuredGrid);
#else
        writer->SetInputData(structuredGrid);
#endif
        writer->Write();



    }

    std::ofstream file;
    char buffer[100];
    sprintf(buffer,"%s_xconcentration.txt",output_name.c_str());
    file.open(buffer);
    const int i = std::floor(((high[0]-low[0])/2-low[0])/ds);
    for (int j=0; j<nx; j++) {
        file << low[1] + (j+0.5)*ds;
        for (int outi=0;outi<=nout;outi++) {
            sprintf(buffer,"%7.7f",output_hist[outi][i*nx+j]*scaleby);
            file << " " << buffer;
        }
        file << std::endl;
    }
    file.close();
    std::cout << std::endl;

    sprintf(buffer,"%s_yconcentration.txt",output_name.c_str());
    file.open(buffer);
    for (int i=0; i<nx; i++) {
        file << low[0] + (i+0.5)*ds;
        for (int outi=0;outi<=nout;outi++) {
            double sum = 0;
            if ((!oned) && (init == 1)) {
                for (int j=0; j<nx; j++) {
                    sum += output_hist[outi][i*nx+j];
                }
                sum /= nx;
            } else {
                const int j = std::floor(((high[1]-low[1])/2-low[1])/ds);
                sum = output_hist[outi][i*nx+j];
            }
            sprintf(buffer,"%7.7f",sum*scaleby);
            file << " " << buffer;
        }
        file << std::endl;
    }
    file.close();
    std::cout << std::endl;

}
