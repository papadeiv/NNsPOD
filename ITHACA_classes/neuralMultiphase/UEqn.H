MRF.correctBoundaryVelocity(U);

tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(rhoPhi, U)
  + MRF.DDt(rho, U)
  //+ turbulence->divDevRhoReff(rho, U)
 ==
    fvOptions(rho, U)
);
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

fvOptions.constrain(UEqn);

if (pimple.momentumPredictor())
{
    solve
    (
        UEqn
     ==
        fvc::reconstruct
        (
            (
                mixture.surfaceTensionForce()
              - ghf*fvc::snGrad(rho)
              - fvc::snGrad(p_rgh)
            ) * mesh.magSf()
        )
    );

    fvOptions.correct(U);
}
