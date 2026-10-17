//! Continuous collision against a *custom* composite target: a `TypedShape::Custom` shape that
//! exposes its parts through `Shape::as_composite_shape` (a game's own static-scene shape, say).
//! The continuous pass culls its BVH and sweeps the culled parts like a `TriMesh`, rather than
//! running one nonlinear cast over the whole shape.

use std::sync::mpsc::channel;

use rapier3d::parry::bounding_volume::{Aabb, BoundingSphere};
use rapier3d::parry::mass_properties::MassProperties;
use rapier3d::parry::partitioning::Bvh;
use rapier3d::parry::query::details::NormalConstraints;
use rapier3d::parry::query::{PointProjection, PointQuery, Ray, RayCast, RayIntersection};
use rapier3d::parry::shape::{
    CompositeShape, Compound, Cylinder, FeatureId, Shape, ShapeType, TriMesh, TriMeshFlags,
    TypedShape,
};
use rapier3d::prelude::*;

/// A parry composite shape reported as `TypedShape::Custom`.
#[derive(Clone)]
struct Custom<S>(S);

impl<S: RayCast> RayCast for Custom<S> {
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        solid: bool,
    ) -> Option<RayIntersection> {
        self.0
            .cast_local_ray_and_get_normal(ray, max_time_of_impact, solid)
    }
}

impl<S: PointQuery> PointQuery for Custom<S> {
    fn project_local_point(&self, pt: Vector, solid: bool) -> PointProjection {
        self.0.project_local_point(pt, solid)
    }

    fn project_local_point_and_get_feature(&self, pt: Vector) -> (PointProjection, FeatureId) {
        self.0.project_local_point_and_get_feature(pt)
    }
}

impl<S: CompositeShape> CompositeShape for Custom<S> {
    fn map_part_at(
        &self,
        shape_id: u32,
        f: &mut dyn FnMut(Option<&Pose>, &dyn Shape, Option<&dyn NormalConstraints>),
    ) {
        self.0.map_part_at(shape_id, f);
    }

    fn bvh(&self) -> &Bvh {
        self.0.bvh()
    }
}

impl<S: Shape + CompositeShape + Clone> Shape for Custom<S> {
    fn compute_local_aabb(&self) -> Aabb {
        self.0.compute_local_aabb()
    }

    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        self.0.compute_local_bounding_sphere()
    }

    fn clone_dyn(&self) -> Box<dyn Shape> {
        Box::new(self.clone())
    }

    fn scale_dyn(&self, _scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        None
    }

    fn mass_properties(&self, density: Real) -> MassProperties {
        self.0.mass_properties(density)
    }

    fn shape_type(&self) -> ShapeType {
        ShapeType::Custom
    }

    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(self)
    }

    fn ccd_thickness(&self) -> Real {
        0.0
    }

    fn ccd_angular_thickness(&self) -> Real {
        core::f32::consts::FRAC_PI_4
    }

    fn as_composite_shape(&self) -> Option<&(dyn CompositeShape + Sync)> {
        Some(self as &(dyn CompositeShape + Sync))
    }
}

struct Harness {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    soft_bodies: SoftBodySet,
    pipeline: PhysicsPipeline,
    bf: BroadPhaseBvh,
    nf: NarrowPhase,
    islands: IslandManager,
    ccd: CCDSolver,
    params: IntegrationParameters,
    gravity: Vector,
}

impl Harness {
    fn new(gravity: Vector) -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            soft_bodies: SoftBodySet::new(),
            pipeline: PhysicsPipeline::new(),
            bf: BroadPhaseBvh::new(),
            nf: NarrowPhase::new(),
            islands: IslandManager::new(),
            ccd: CCDSolver::new(),
            params: IntegrationParameters::default(),
            gravity,
        }
    }

    fn step_with(&mut self, events: &dyn EventHandler) {
        self.pipeline.step(
            self.gravity,
            &self.params,
            &mut self.islands,
            &mut self.bf,
            &mut self.nf,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.soft_bodies,
            &mut self.ccd,
            &(),
            events,
        );
    }

    fn step(&mut self) {
        self.step_with(&());
    }

    fn translation(&self, h: RigidBodyHandle) -> Vector {
        self.bodies[h].translation()
    }

    /// A 2 cm `ccd_enabled` cube.
    fn insert_small_cube(&mut self, translation: Vector, linvel: Vector) -> RigidBodyHandle {
        let body = self.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(translation)
                .linvel(linvel)
                .ccd_enabled(true),
        );
        self.colliders.insert_with_parent(
            ColliderBuilder::cuboid(0.01, 0.01, 0.01),
            body,
            &mut self.bodies,
        );
        body
    }
}

/// How the static ground or wall is represented.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Target {
    Cuboid,
    TriMesh,
    Custom,
}

/// A flat `TriMesh` at `y = 0` over `[-half, half]²`, `cells × cells` quads, facing up.
fn grid_floor(half: Real, cells: u32) -> TriMesh {
    let step = 2.0 * half / cells as Real;
    let mut vertices = Vec::new();
    for i in 0..=cells {
        for j in 0..=cells {
            vertices.push(Vector::new(
                -half + i as Real * step,
                0.0,
                -half + j as Real * step,
            ));
        }
    }
    let id = |i: u32, j: u32| i * (cells + 1) + j;
    let mut indices = Vec::new();
    for i in 0..cells {
        for j in 0..cells {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            indices.push([a, c, b]);
            indices.push([a, d, c]);
        }
    }
    TriMesh::with_flags(vertices, indices, TriMeshFlags::FIX_INTERNAL_EDGES).unwrap()
}

/// A 20 m × 20 m floor with its top at `y = 0`.
fn insert_floor(h: &mut Harness, target: Target) {
    let builder = match target {
        Target::Cuboid => {
            ColliderBuilder::cuboid(10.0, 0.5, 10.0).translation(Vector::new(0.0, -0.5, 0.0))
        }
        Target::TriMesh => ColliderBuilder::new(SharedShape::new(grid_floor(10.0, 16))),
        Target::Custom => ColliderBuilder::new(SharedShape::new(Custom(grid_floor(10.0, 16)))),
    };
    h.colliders.insert(builder);
}

/// A two-triangle wall in the `x = 0` plane.
fn wall() -> TriMesh {
    let vertices = vec![
        Vector::new(0.0, -5.0, -5.0),
        Vector::new(0.0, 5.0, -5.0),
        Vector::new(0.0, 5.0, 5.0),
        Vector::new(0.0, -5.0, 5.0),
    ];
    TriMesh::new(vertices, vec![[0, 1, 2], [0, 2, 3]]).unwrap()
}

/// The custom composite target sweeps exactly like the equivalent `TriMesh`: the impact step
/// of a fast cube (CCD only — no contact exists before it) lands on the same pose.
#[test]
fn custom_composite_impact_matches_trimesh() {
    let run = |target: Target| {
        let mut h = Harness::new(Vector::new(0.0, -9.81, 0.0));
        insert_floor(&mut h, target);
        let cube = h.insert_small_cube(Vector::new(0.3, 1.0, -0.2), Vector::new(3.0, -40.0, 2.0));
        (0..2)
            .map(|_| {
                h.step();
                h.translation(cube)
            })
            .collect::<Vec<_>>()
    };
    let trimesh = run(Target::TriMesh);
    let custom = run(Target::Custom);
    assert!(
        trimesh[1].y < 0.1,
        "the second step should end on the floor (y = {})",
        trimesh[1].y
    );
    for (step, (a, b)) in trimesh.iter().zip(&custom).enumerate() {
        assert!(
            (*a - *b).length() < 1.0e-5,
            "step {step}: custom composite {b:?} vs trimesh {a:?}"
        );
    }
}

/// A fast small cube does not tunnel through a custom composite floor, and settles on it.
#[test]
fn fast_cube_does_not_tunnel_custom_floor() {
    let mut h = Harness::new(Vector::new(0.0, -9.81, 0.0));
    insert_floor(&mut h, Target::Custom);
    let cube = h.insert_small_cube(Vector::new(0.0, 1.0, 0.0), Vector::new(1.0, -40.0, 0.5));
    h.bodies[cube].set_angvel(Vector::new(20.0, 5.0, -10.0), true);

    for i in 0..180 {
        h.step();
        let y = h.translation(cube).y;
        assert!(
            y > 0.0,
            "cube tunneled through the floor at step {i} (y = {y})"
        );
    }
    let y = h.translation(cube).y;
    assert!(
        y > 0.005 && y < 0.02,
        "cube did not settle on the floor (y = {y})"
    );
}

/// A fast small cube stops at a custom composite wall from either side (two-sided).
#[test]
fn fast_cube_does_not_tunnel_custom_wall() {
    for side in [-1.0, 1.0] {
        let mut h = Harness::new(Vector::ZERO);
        h.colliders
            .insert(ColliderBuilder::new(SharedShape::new(Custom(wall()))));
        let cube = h.insert_small_cube(
            Vector::new(side, 0.3, 0.1),
            Vector::new(-side * 40.0, 0.0, 0.0),
        );
        for i in 0..60 {
            h.step();
            let x = h.translation(cube).x;
            assert!(
                x * side > 0.0,
                "cube tunneled through the wall from x = {side} at step {i} (x = {x})"
            );
        }
    }
}

/// A thin `ccd_enabled` shard pushed along a custom composite floor slides exactly as on the
/// equivalent `TriMesh` and nearly as far as on a cuboid floor, then rests without jitter. It is
/// never pinned by its own resting contact; as on a `TriMesh`, its sweep only stops it at an
/// internal edge it reaches within a step (the rest of that step's motion is lost, its velocity
/// kept), hence the shorter slide than on the single cuboid.
#[test]
fn sliding_shard_is_not_pinned_by_custom_floor() {
    let run = |target: Target| {
        let mut h = Harness::new(Vector::new(0.0, -9.81, 0.0));
        insert_floor(&mut h, target);
        let shard = h.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(-4.3, 0.005, 0.37))
                .ccd_enabled(true),
        );
        h.colliders.insert_with_parent(
            ColliderBuilder::cuboid(0.05, 0.005, 0.05),
            shard,
            &mut h.bodies,
        );
        // Settle, then push it along the floor.
        for _ in 0..30 {
            h.step();
        }
        let start = h.translation(shard);
        h.bodies[shard].set_linvel(Vector::new(6.0, 0.0, 1.5), true);
        let mut rest_ys = Vec::new();
        for i in 0..240 {
            h.step();
            let y = h.translation(shard).y;
            assert!(
                y > 0.0,
                "{target:?}: shard sank through the floor at step {i} (y = {y})"
            );
            if i >= 180 {
                rest_ys.push(y);
            }
        }
        let rb = &h.bodies[shard];
        let travelled = (h.translation(shard) - start).length();
        let spread = rest_ys
            .iter()
            .fold(0.0, |m: Real, y| m.max((y - rest_ys[0]).abs()));
        (
            travelled,
            rb.linvel().length(),
            rb.angvel().length(),
            spread,
        )
    };

    let cuboid = run(Target::Cuboid);
    let trimesh = run(Target::TriMesh);
    let custom = run(Target::Custom);
    for (target, (travelled, linvel, angvel, spread)) in [
        (Target::Cuboid, cuboid),
        (Target::TriMesh, trimesh),
        (Target::Custom, custom),
    ] {
        assert!(
            linvel < 1.0e-2 && angvel < 1.0e-2 && spread < 1.0e-3,
            "{target:?}: shard not at rest (linvel {linvel}, angvel {angvel}, y spread {spread})"
        );
        assert!(
            travelled > 2.0,
            "{target:?}: shard pinned (travelled {travelled} m)"
        );
    }
    assert!(
        (custom.0 - trimesh.0).abs() < 1.0e-4,
        "custom floor slide {} m vs trimesh {} m",
        custom.0,
        trimesh.0
    );
    assert!(
        custom.0 > 0.85 * cuboid.0,
        "custom floor slide {} m vs cuboid {} m",
        custom.0,
        cuboid.0
    );
}

/// A custom composite with a part that has no point-cloud proxy (a cylinder) falls back to the
/// whole-collider nonlinear cast, which still stops the fast cube.
#[test]
fn custom_composite_without_proxy_parts_falls_back() {
    let mut h = Harness::new(Vector::new(0.0, -9.81, 0.0));
    let disk = Compound::new(vec![(
        Pose::from_translation(Vector::new(0.0, -0.05, 0.0)),
        SharedShape::new(Cylinder::new(0.05, 10.0)),
    )]);
    h.colliders
        .insert(ColliderBuilder::new(SharedShape::new(Custom(disk))));
    let cube = h.insert_small_cube(Vector::new(0.0, 1.0, 0.0), Vector::new(0.0, -40.0, 0.0));
    for i in 0..120 {
        h.step();
        let y = h.translation(cube).y;
        assert!(
            y > 0.0,
            "cube tunneled through the disk at step {i} (y = {y})"
        );
    }
}

/// A fast ball crossing a custom composite sensor within one step (never overlapping it at a
/// step boundary) still reports the crossing.
#[test]
fn fast_ball_crossing_custom_sensor_reports_events() {
    let mut h = Harness::new(Vector::ZERO);
    let sensor = h.colliders.insert(
        ColliderBuilder::new(SharedShape::new(Custom(wall())))
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS),
    );
    let ball = h.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(-1.0, 0.0, 0.0))
            .linvel(Vector::new(120.0, 0.0, 0.0))
            .ccd_enabled(true),
    );
    h.colliders
        .insert_with_parent(ColliderBuilder::ball(0.05), ball, &mut h.bodies);

    let (collision_send, collision_recv) = channel();
    let (force_send, _force_recv) = channel();
    let (tear_send, _tear_recv) = channel();
    let events = ChannelEventCollector::new(collision_send, force_send, tear_send);
    h.step_with(&events);

    assert!(
        h.translation(ball).x > 0.5,
        "the ball should pass the sensor"
    );
    let received: Vec<CollisionEvent> = collision_recv.try_iter().collect();
    let started = received
        .iter()
        .any(|e| e.started() && (e.collider1() == sensor || e.collider2() == sensor));
    let stopped = received
        .iter()
        .any(|e| e.stopped() && (e.collider1() == sensor || e.collider2() == sensor));
    assert!(started && stopped, "missing sensor events: {received:?}");
}
