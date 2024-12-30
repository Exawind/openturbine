use std::cmp;

use faer::{col, unzipped, zipped, Col, ColMut, ColRef, Mat, MatMut};
use itertools::Itertools;

use crate::{
    node::NodeFreedomMap,
    state::State,
    util::vec_tilde,
    util::{
        axial_vector_of_matrix, matrix_ax, quat_as_matrix, quat_inverse, quat_rotate_vector, Quat,
    },
};

#[derive(Clone, Copy)]
pub enum ConstraintKind {
    Rigid,
    Prescribed,
}

pub struct ConstraintInput {
    pub id: usize,
    pub kind: ConstraintKind,
    pub node_id_base: usize,
    pub node_id_target: usize,
    pub x0: Col<f64>,
    pub vec: Col<f64>,
}

pub struct Constraints {
    pub n_dofs: usize,
    pub constraints: Vec<Constraint>,
}

impl Constraints {
    pub fn new(inputs: &[ConstraintInput], nfm: &NodeFreedomMap) -> Self {
        let mut n_dofs = 0;
        let constraints = inputs
            .iter()
            .map(|inp| {
                let c = Constraint::new(n_dofs, inp, nfm);
                n_dofs += c.n_dofs;
                c
            })
            .collect_vec();
        Self {
            n_dofs,
            constraints,
        }
    }

    pub fn assemble_constraints(
        &mut self,
        nfm: &NodeFreedomMap,
        state: &State,
        mut phi: ColMut<f64>,
        mut b: MatMut<f64>,
    ) {
        // Loop through constraints and calculate residual and gradient
        self.constraints.iter_mut().for_each(|c| {
            // Get base and target node
            let u_base = state.u.col(c.node_id_base).subrows(0, 3);
            let r_base = state.u.col(c.node_id_base).subrows(3, 4);
            let u_target = state.u.col(c.node_id_target).subrows(0, 3);
            let r_target = state.u.col(c.node_id_target).subrows(3, 4);

            // Switch calculation based on constraint type
            match c.kind {
                ConstraintKind::Prescribed => c.calculate_prescribed(u_target, r_target),
                ConstraintKind::Rigid => c.calculate_rigid(u_base, r_base, u_target, r_target),
            }
        });

        // Assemble residual and gradient into global matrix and array
        self.constraints.iter_mut().for_each(|c| {
            // Subsection of residual for this constraint
            let mut phi_c = phi.as_mut().subrows_mut(c.first_dof_index, c.phi.nrows());

            // Assemble residual and gradient based on type
            let dofs_target = &nfm.node_dofs[c.node_id_target];
            let mut b_target = b.as_mut().submatrix_mut(
                c.first_dof_index,
                dofs_target.first_dof_index,
                c.b_target.nrows(),
                c.b_target.ncols(),
            );
            match c.kind {
                ConstraintKind::Prescribed => {
                    phi_c.copy_from(&c.phi);
                    b_target.copy_from(&c.b_target);
                }
                _ => {
                    phi_c.copy_from(&c.phi);
                    b_target.copy_from(&c.b_target);
                    let dofs_base = &nfm.node_dofs[c.node_id_base];
                    let mut b_base = b.as_mut().submatrix_mut(
                        c.first_dof_index,
                        dofs_base.first_dof_index,
                        c.b_base.nrows(),
                        c.b_base.ncols(),
                    );
                    b_base.copy_from(&c.b_base);
                }
            }
        });
    }
}

pub struct Constraint {
    kind: ConstraintKind,
    first_dof_index: usize,
    n_dofs: usize,
    node_id_base: usize,
    node_id_target: usize,
    x0: Col<f64>,
    phi: Col<f64>,
    b_base: Mat<f64>,
    b_target: Mat<f64>,
    u_prescribed: Col<f64>,
    rbinv: Col<f64>,
    rt_rbinv: Col<f64>,
    r_x0: Col<f64>,
    /// Rotation matrix `[3,3]`
    c: Mat<f64>,
    /// Axial vector of rotation matrix
    ax: Mat<f64>,
}

impl Constraint {
    fn new(first_dof_index: usize, input: &ConstraintInput, nfm: &NodeFreedomMap) -> Self {
        let n_dofs_base = nfm.node_dofs[input.node_id_base].n_dofs;
        let n_dofs_target = nfm.node_dofs[input.node_id_target].n_dofs;

        // Get number of active DOFs in constraint
        let n_dofs = match input.kind {
            ConstraintKind::Prescribed => n_dofs_target,
            ConstraintKind::Rigid => cmp::min(n_dofs_base, n_dofs_target),
        };

        Self {
            kind: input.kind,
            first_dof_index,
            n_dofs,
            node_id_base: input.node_id_base,
            node_id_target: input.node_id_target,
            x0: input.x0.clone(),
            u_prescribed: col![0., 0., 0., 1., 0., 0., 0.],
            phi: Col::<f64>::zeros(n_dofs),
            b_base: -1. * Mat::<f64>::identity(n_dofs, n_dofs_base),
            b_target: Mat::<f64>::identity(n_dofs, n_dofs_target),
            r_x0: Col::<f64>::zeros(3),
            rbinv: Col::<f64>::zeros(4),
            rt_rbinv: Col::<f64>::zeros(4),
            c: Mat::<f64>::zeros(3, 3),
            ax: Mat::zeros(3, 3),
        }
    }

    pub fn set_displacement(&mut self, x: f64, y: f64, z: f64, rx: f64, ry: f64, rz: f64) {
        let mut q = col![0., 0., 0., 0.];
        q.as_mut()
            .quat_from_rotation_vector(col![rx, ry, rz].as_ref());
        self.u_prescribed[0] = x;
        self.u_prescribed[1] = y;
        self.u_prescribed[2] = z;
        self.u_prescribed[3] = q[0];
        self.u_prescribed[4] = q[1];
        self.u_prescribed[5] = q[2];
        self.u_prescribed[6] = q[3];
    }

    fn calculate_prescribed(&mut self, u_target: ColRef<f64>, r_target: ColRef<f64>) {
        let u_base = self.u_prescribed.subrows(0, 3);
        let r_base = self.u_prescribed.subrows(3, 4);

        // Position residual: Phi(0:3) = u2 - u1
        self.phi[0] = u_target[0] - u_base[0];
        self.phi[1] = u_target[1] - u_base[1];
        self.phi[2] = u_target[2] - u_base[2];

        if self.n_dofs == 3 {
            return;
        }

        // Angular residual:  Phi(3:6) = axial(R2*inv(R1))
        quat_inverse(r_base, self.rbinv.as_mut());
        self.rt_rbinv
            .as_mut()
            .quat_compose(r_target.as_ref(), self.rbinv.as_ref());
        quat_as_matrix(self.rt_rbinv.as_ref(), self.c.as_mut());
        axial_vector_of_matrix(self.c.as_ref(), self.phi.subrows_mut(3, 3));

        // Constraint Gradient
        // Set at initialization B(0:3,0:3) = I
        // B(3:6,3:6) = AX(R1*inv(R2)) = transpose(AX(R2*inv(R1)))
        matrix_ax(
            self.c.as_ref(),
            self.b_target.submatrix_mut(3, 3, 3, 3).transpose_mut(),
        );
    }

    fn calculate_rigid(
        &mut self,
        u_base: ColRef<f64>,
        r_base: ColRef<f64>,
        u_target: ColRef<f64>,
        r_target: ColRef<f64>,
    ) {
        // Position residual: Phi(0:3) = u2 + X0 - u1 - R1*X0
        quat_rotate_vector(r_base.as_ref(), self.x0.as_ref(), self.r_x0.as_mut());
        zipped!(
            &mut self.phi.subrows_mut(0, 3),
            &u_base,
            &u_target,
            &self.x0,
            &self.r_x0
        )
        .for_each(|unzipped!(mut phi, u1, u2, x0, rb_x0)| *phi = *u2 + *x0 - *u1 - *rb_x0);

        // Angular residual:  Phi(3:6) = axial(R2*inv(rb))
        if self.n_dofs == 6 {
            quat_inverse(r_base, self.rbinv.as_mut());
            self.rt_rbinv
                .as_mut()
                .quat_compose(r_target, self.rbinv.as_ref());
            quat_as_matrix(self.rt_rbinv.as_ref(), self.c.as_mut());
            axial_vector_of_matrix(self.c.as_ref(), self.phi.subrows_mut(3, 3));
        }

        // Base constraint gradient
        // Set at initialization B(0:3,0:3) = -I
        // B(0:3,3:6) = tilde(R1*X0)
        vec_tilde(self.r_x0.as_ref(), self.b_base.submatrix_mut(0, 3, 3, 3));
        if self.n_dofs == 6 {
            // AX(c)
            matrix_ax(self.c.as_ref(), self.ax.as_mut());
            // B(3:6,3:6) = -AX(R2*inv(R1))
            zipped!(&mut self.b_base.submatrix_mut(3, 3, 3, 3), &self.ax)
                .for_each(|unzipped!(mut b, ax)| *b = -*ax);
        }

        // Target constraint gradient
        // Set at initialization B(0:3,0:3) = I
        if self.n_dofs == 6 {
            // B(3:6,3:6) = transpose(AX(R2*inv(R1)))
            self.b_target
                .submatrix_mut(3, 3, 3, 3)
                .copy_from(self.ax.transpose());
        }
    }
}
