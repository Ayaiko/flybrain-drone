// Offline brushed-quad endurance experiment. Units are SI except thrust in gf.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
using Axes=std::array<double,3>;
constexpr double dt=.004, g=9.81, arm=.0403051;
constexpr int mixer[4][3]={{1,1,1},{-1,1,-1},{-1,-1,1},{1,-1,-1}};
// Mission axes: x right, y forward, z up. Simulated velocity aiding is assumed.
struct LightTrial {
    int read_us;
    bool scheduled;
    double detour_m;
};
struct Run {
    int adc_samples=0, adc_last=0, missed=0;
    double altitude=0, climb=0, gap=9, time=0, stopgap=-1;
    const char* outcome="timeout";
};
enum Stage { Takeoff, Approach, Brake, Sidestep, Pass };
double ray(double x,double y,double dx,double dy) {
    double lo=0,hi=2;
    for(auto bounds : {std::array<double,4>{x,dx,-.19,.19},{y,dy,1.18,1.60}}){
        auto [p,d,a,b]=bounds;
        if(std::abs(d)<1e-8){if(p<a || p>b)return 2;continue;}
        double t1=(a-p)/d,t2=(b-p)/d;
        lo=std::max(lo,std::min(t1,t2));hi=std::min(hi,std::max(t1,t2));
    }
    return lo<=hi?lo:2;
}
Run trial(int seed,bool revised,bool degraded,FILE* trace, const LightTrial* light=nullptr){
    constexpr double mass=62,gf=18,seconds=75;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> noise(-1.,1.);
    Axes angle{}, rate{}, estimate{}, integral{}, error{}, adjust{};
    std::array<double,4> gain, motor{}, duty{};
    for(auto &v:gain) v=1+.04*noise(rng);
    double lift_bias=0, height_obs=0, last_height=0, climb_obs=0;
    double height_i=0;
    Run r;
    int adc_remaining=0;
    Stage stage=Takeoff;
    double x_est=0,y_est=0,refx=0,refy=0;
    double x=.05*noise(rng), y=0,vx=0,vy=0,disp=0,stop_time=0;
    double bias_sum=0, bias_est=0;
    double range=0, age=1, queued=0; bool pending=false;
    const double vbias=.006*noise(rng);

    for(int k=0;k<static_cast<int>(seconds/dt);++k){
        double t=k*dt;
        bool miss_control=false;
        if(light){
            // Synthetic end-to-end ADC task cost, including application overhead.
            // Priority mode reserves 3000 us of the 4000 us frame for flight.
            if(k%25==0) adc_remaining=light->read_us;
            if(adc_remaining>0){
                miss_control=!light->scheduled && adc_remaining>1000;
                adc_remaining=std::max(0,adc_remaining-(light->scheduled?1000:4000));
                if(adc_remaining==0){
                    r.adc_last=int(2048+1400*std::sin(.8*t));
                    ++r.adc_samples;
                }
            }
            if(miss_control) ++r.missed;
        }
        // Identical disturbance/noise sequence for each controller on a given seed.
        const double raw_vx=vx+vbias+.01*noise(rng);
        if(revised && k<100){bias_sum+=raw_vx; bias_est=bias_sum/(k+1);}
        const double sensed_vx=raw_vx-bias_est, sensed_vy=vy+.01*noise(rng);
        double measurement_noise=.012*noise(rng), drop=noise(rng);
        age+=dt;
        if(k%25==15 && pending){range=queued;age=.06;pending=false;}
        if(k%25==0 && drop>-.90){
            queued=std::clamp(ray(x,y,std::sin(angle[2]),std::cos(angle[2]))+measurement_noise,.03,2.);
            pending=true;
        }
        double cmdx=0,cmdy=0;
        // Begin the obstacle course only after the full 40-second thrust loss.
        if(stage==Takeoff && t>=40 && height_obs>.52) stage=Approach;
        if(stage==Approach && age<.25){
            if(range<(revised?.62:.44)){stage=Brake; refx=x_est;refy=y_est;}
            else cmdy=.32;
        }
        if(stage==Brake){
            stop_time=(std::hypot(sensed_vx,sensed_vy)<.045)?stop_time+dt:0;
            if(stop_time>.15){stage=Sidestep;r.stopgap=1.18-y-.091;}
        }
        if(stage==Sidestep){cmdx=.22;disp+=sensed_vx*dt;if(disp>(light?light->detour_m:(revised?.50:.40)))stage=Pass;}
        if(stage==Pass && age<.25 && range>.28)cmdy=.32;
        if(revised){
            // Dead reckoning in the onboard estimate, no access to box coordinates.
            x_est+=dt*sensed_vx;y_est+=dt*sensed_vy;
            refx+=dt*cmdx;refy+=dt*cmdy;
            cmdx+=std::clamp((refx-x_est)*1.2,-.24,.24);
            cmdy+=std::clamp((refy-y_est)*1.2,-.24,.24);
        }
        double ax=std::clamp((cmdx-sensed_vx)*2.4,-1.,1.);
        double ay=std::clamp((cmdy-sensed_vy)*2.4,-1.,1.);
        double sy=std::sin(estimate[2]),cy=std::cos(estimate[2]);
        Axes target_angle={std::atan((ax*cy-ay*sy)/g),std::atan((ax*sy+ay*cy)/g),0};
        Axes gyro;
        for(int j=0;j<3;++j){
            gyro[j]=rate[j]+.006*noise(rng)+(j==2?.001:0.);
            estimate[j]+=dt*gyro[j];
            double accel_angle=angle[j]+.012*noise(rng);
            if(j<2) estimate[j]+=.018*(accel_angle-estimate[j]);
        }
        if(k%10==0){
            double alpha=.42;
            height_obs+=alpha*(r.altitude+.025*noise(rng)-height_obs);
            double beta=.38;
            climb_obs+=beta*((height_obs-last_height)/.04-climb_obs);
            last_height=height_obs;
        }
        if(k%3750==1250) rate[seed%3]+=.65;
        // On a missed deadline, retain the previous four motor commands.
        if(!miss_control){
            double target=std::min(.65, t*.18);
            double dz=target-height_obs;
            height_i=std::clamp(height_i+dz*dt,-1.8,1.8);
            lift_bias=(3*dz-2.6*climb_obs+1.6*height_i)/g;
            double base=mass/(4*gf)*(1+lift_bias)*std::min(1.,t/.45);
            for(int j=0;j<3;++j){
                error[j]=target_angle[j]-estimate[j];
                integral[j]=std::clamp(integral[j]+error[j]*dt,-.25,.25);
                double accel=55*error[j]-12*gyro[j]+42*integral[j];
                double moment_arm=j==2?.006:arm;
                double inertia=mass*.001444*(j==2?1.8:1.);
                adjust[j]=inertia*accel/(4*moment_arm*gf*g);
            }
            // Preserve differential control headroom at high collective.
            if(revised){
                double peak=0;
                for(auto &row:mixer)peak=std::max(peak,row[0]*adjust[0]+row[1]*adjust[1]+row[2]*adjust[2]);
                base=std::min(base,1-std::min(.12,peak));
            }
            for(int i=0;i<4;++i){
                double u=base;
                for(int j=0;j<3;++j) u+=mixer[i][j]*adjust[j];
                duty[i]=(revised && k<100)?0:std::clamp(u,0.,1.);
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
        if(r.altitude>0){
            double right=force/(mass/1000)*std::sin(angle[0]);
            double forward=force/(mass/1000)*std::sin(angle[1])*std::cos(angle[0]);
            double c=std::cos(angle[2]), sn=std::sin(angle[2]);
            vx+=dt*(right*c+forward*sn-.12*vx);
            vy+=dt*(forward*c-right*sn-.12*vy);
            x+=dt*vx;y+=dt*vy;
        }
        double gap=std::hypot(std::max({-.19-x,0.,x-.19}),std::max({1.18-y,0.,y-1.60}))-.091;
        r.gap=std::min(r.gap,gap);r.time=t;
        if(gap<=0)r.outcome="contact";
        else if(t>=6 && (r.altitude<.4 || r.altitude>.9))r.outcome="altitude";
        else if(std::abs(x)+.091>=1)r.outcome="wall";
        else if(stage==Pass && y>=2.15 && r.stopgap>0)r.outcome="finish";
        bool end=r.outcome[0]!='t' || k+1==int(seconds/dt);
        if(trace && (k%25==0 || end))fprintf(trace,"%.3f,%.4f,%.4f,%.4f,%d,%s\n",t,x,y,r.altitude,stage,r.outcome);
        if(end)break;
    }
    return r;
}
int light_trials(){
    puts("seed,phase,loss_pct,mode,task_us,detour_m,result,time_s,min_gap_m,stop_gap_m,adc_samples,missed_controls");
    for(int seed=400;seed<488;++seed) for(bool loss:{false,true})
        for(int mode=0;mode<4;++mode) for(int cost:{250,6000}){
            LightTrial light{cost,mode>=2,mode==3?.56:.50};
            Run r=trial(seed,true,loss,nullptr,mode==0?nullptr:&light);
            printf("%d,%s,%d,%d,%d,%.2f,%s,%.3f,%.4f,%.4f,%d,%d\n",
                   seed,seed<424?"development":"holdout",loss?18:5,mode,cost,light.detour_m,
                   r.outcome,r.time,r.gap,r.stopgap,r.adc_samples,r.missed);
        }
    return 0;
}
int main(int argc,char** argv){
    if(argc>1 && std::string(argv[1])=="light")return light_trials();
    bool validate=argc>1; (void)argv;
    puts("phase,seed,loss_pct,version,result,time_s,min_gap_m,stop_gap_m");
    for(int seed=validate?300:0;seed<(validate?364:24);++seed)
      for(bool loss:{false,true})for(bool version:{false,true}){
        FILE* trace=nullptr;char path[80];
        if(seed==302){
            snprintf(path,sizeof(path),"trace-%d-%d.csv",loss?18:5,int(version));
            trace=fopen(path,"w"); if(!trace)return 1;
            fputs("time,x,y,z,stage,status\n",trace);
        }
        Run r=trial(seed,version,loss,trace);
        if(trace)fclose(trace);
        printf("%s,%d,%d,%s,%s,%.3f,%.4f,%.4f\n",validate?"validation":"development",seed,loss?18:5,version?"after":"before",r.outcome,r.time,r.gap,r.stopgap);
      }
}
