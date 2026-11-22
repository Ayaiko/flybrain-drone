// Offline brushed-quad endurance experiment. Units are SI except thrust in gf.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
using Axes=std::array<double,3>;
constexpr double dt=.004, g=9.81, arm=.0403051;
constexpr int mixer[4][3]={{1,1,1},{-1,1,-1},{-1,-1,1},{1,-1,-1}};
struct Run {
    double altitude=0, climb=0, min_height=10, max_error=0, peak_tilt=0;
    int missed=0;
    const char* outcome="pass";
};
Run trial(int seed, double mass, double gf, bool tuned, double seconds, bool degraded){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> noise(-1.,1.);
    Axes angle{}, rate{}, estimate{}, integral{}, error{}, adjust{};
    std::array<double,4> gain, motor{}, duty{};
    for(auto &v:gain) v=1+.04*noise(rng);
    double lift_bias=0, height_obs=0, last_height=0, climb_obs=0;
    double height_i=0;
    Run r;
    for(int k=0;k<static_cast<int>(seconds/dt);++k){
        double t=k*dt;
        // Identical disturbance/noise sequence for each controller on a given seed.
        Axes gyro;
        for(int j=0;j<3;++j){
            gyro[j]=rate[j]+.006*noise(rng)+(j==2?.001:0.);
            estimate[j]+=dt*gyro[j];
            double accel_angle=angle[j]+.012*noise(rng);
            if(j<2) estimate[j]+=.018*(accel_angle-estimate[j]);
        }
        if(k%10==0){
            double alpha=tuned?.42:.14;
            height_obs+=alpha*(r.altitude+.025*noise(rng)-height_obs);
            double beta=tuned?.38:.18;
            climb_obs+=beta*((height_obs-last_height)/.04-climb_obs);
            last_height=height_obs;
        }
        if(k%3750==1250) rate[seed%3]+=.65;
        // Synthetic bus occupancy: a 40 ms transaction every 100 ms.
        // Cooperative servicing assumes each slice fits before the control deadline.
        if(!tuned && k%25<10) ++r.missed;
        else {
            double target=std::min(.65, t*.18);
            double dz=target-height_obs;
            if(tuned) height_i=std::clamp(height_i+dz*dt,-1.8,1.8);
            lift_bias=(tuned ? 3*dz-2.6*climb_obs+1.6*height_i : 5*dz-3*climb_obs)/g;
            double base=mass/(4*gf)*(1+lift_bias)*std::min(1.,t/.45);
            for(int j=0;j<3;++j){
                error[j]=-estimate[j];
                integral[j]=std::clamp(integral[j]+error[j]*dt,-.25,.25);
                double accel=55*error[j]-12*gyro[j]+42*integral[j];
                double moment_arm=j==2?.006:arm;
                double inertia=mass*.001444*(j==2?1.8:1.);
                adjust[j]=inertia*accel/(4*moment_arm*gf*g);
            }
            for(int i=0;i<4;++i){
                double u=base;
                for(int j=0;j<3;++j) u+=mixer[i][j]*adjust[j];
                duty[i]=std::clamp(u,0.,1.);
            }
        }
        double force=0;
        Axes torque{};
        for(int i=0;i<4;++i){
            motor[i]+=(duty[i]-motor[i])*(1-std::exp(-dt/.035));
            // Sensitivity only, not a fitted battery discharge model.
            double sag=1-(degraded?.18:.05)*std::min(1.,t/40.);
            double f=motor[i]*gain[i]*gf*sag*g/1000;
            force+=f;
            for(int j=0;j<3;++j) torque[j]+=f*mixer[i][j]*(j==2?.006:arm);
        }
        double az=force*std::cos(angle[0])*std::cos(angle[1])/(mass/1000)-g;
        r.climb+=az*dt;
        r.altitude+=r.climb*dt;
        if(r.altitude<=0){r.altitude=0;r.climb=std::max(0.,r.climb);}
        if(r.altitude>0)
            for(int j=0;j<3;++j){
                rate[j]+=dt*torque[j]/(mass/1000*.001444*(j==2?1.8:1.));
                angle[j]+=dt*rate[j];
            }
        if(t>=6){
            r.min_height=std::min(r.min_height,r.altitude);
            r.max_error=std::max(r.max_error,std::abs(.65-r.altitude));
            r.peak_tilt=std::max({r.peak_tilt,std::abs(angle[0]),std::abs(angle[1])});
        }
        if(t>=6 && (r.altitude<.4 || r.altitude>.9 || r.peak_tilt>.35)){
            r.outcome="envelope";break;
        }
    }
    if(r.min_height>=.4 && r.max_error<=.25 && r.peak_tilt<=.35 && (r.max_error>.15 || r.peak_tilt>.16)) r.outcome="outside_target";
    return r;
}
int main(){
    puts("seed,phase,mass_g,motor_gf,thrust_loss_pct,controller,scheduled_s,result,min_height_m,max_height_error_m,max_tilt_rad,missed_updates");
    for(int seed=0;seed<60;++seed) for(int mass:{56,62}) for(int gf:{16,18,22})
        for(bool degraded:{false,true}) for(bool tuned:{false,true}){
            double duration=seed%6==5?180:60;
            Run r=trial(seed,mass,gf,tuned,duration,degraded);
            printf("%d,%s,%d,%d,%d,%s,%.0f,%s,%.3f,%.3f,%.3f,%d\n",seed,seed<36?"development":"holdout",mass,gf,degraded?18:5,tuned?"revised":"blocking",duration,r.outcome,r.min_height,r.max_error,r.peak_tilt,r.missed);
        }
}
