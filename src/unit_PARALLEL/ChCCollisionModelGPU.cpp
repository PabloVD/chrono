//////////////////////////////////////////////////
//  
//   ChCModelGPU.cpp
//
// ------------------------------------------------
// 	 Copyright:Alessandro Tasora / DeltaKnowledge
//             www.deltaknowledge.com 
// ------------------------------------------------
///////////////////////////////////////////////////

#ifndef CH_NOCUDA 

#include "ChCCollisionModelGPU.h" 
#include "physics/ChBody.h"
#include "physics/ChSystem.h"

namespace chrono {
	namespace collision {

		ChCollisionModelGPU::ChCollisionModelGPU() {
			nObjects = 0;
			colFam = -1;
			noCollWith = -2;
		}

		ChCollisionModelGPU::~ChCollisionModelGPU() {
			mData.clear();
		}
		int ChCollisionModelGPU::ClearModel() {

			if (GetPhysicsItem()->GetSystem() && GetPhysicsItem()->GetCollide()) {
				GetPhysicsItem()->GetSystem()->GetCollisionSystem()->Remove(this);
			}
			mData.clear();
			nObjects = 0;
			colFam = -1;
			noCollWith = -2;
			return 1;
		}

		int ChCollisionModelGPU::BuildModel() {
			if (GetPhysicsItem()->GetSystem() && GetPhysicsItem()->GetCollide()) {
				GetPhysicsItem()->GetSystem()->GetCollisionSystem()->Add(this);
			}
			return 1;
		}
		bool ChCollisionModelGPU::AddSphere(double radius, ChVector<>* pos) {
			double mass = this->GetBody()->GetMass();
			this->GetBody()->SetInertiaXX(ChVector<> (2 / 5.0 * mass * radius * radius, 2 / 5.0 * mass * radius * radius, 2 / 5.0 * mass * radius * radius));
			model_type = SPHERE;
			nObjects++;
			bData tData;
			tData.A = make_float3(pos->x, pos->y, pos->z);
			tData.B = make_float3(radius, 0, 0);
			tData.C = make_float3(0, 0, 0);
			tData.R = make_float4(1,0,0,0);
			tData.type = SPHERE;
			mData.push_back(tData);
			return true;
		}
		bool ChCollisionModelGPU::AddEllipsoid(double rx, double ry, double rz, ChVector<>* pos, ChMatrix33<>* rot) {
			double mass = this->GetBody()->GetMass();
			this->GetBody()->SetInertiaXX(ChVector<> (1 / 5.0 * mass * (ry * ry + rz * rz), 1 / 5.0 * mass * (rx * rx + rz * rz), 1 / 5.0 * mass * (rx * rx + ry * ry)));
			model_type = ELLIPSOID;
			nObjects++;
			bData tData;
			tData.A = make_float3(pos->x, pos->y, pos->z);
			tData.B = make_float3(rx, ry, rz);
			tData.C = make_float3(0, 0, 0);
			tData.R = make_float4(rot->Get_A_quaternion().e0, rot->Get_A_quaternion().e1, rot->Get_A_quaternion().e2, rot->Get_A_quaternion().e3);
			tData.type = ELLIPSOID;
			mData.push_back(tData);
			return true;
		}
		bool ChCollisionModelGPU::AddBox(double hx, double hy, double hz, ChVector<>* pos, ChMatrix33<>* rot) {

			double mass = this->GetBody()->GetMass();
			this->GetBody()->SetInertiaXX(ChVector<> (1 / 12.0 * mass * (hy * hy + hz * hz), 1 / 12.0 * mass * (hx * hx + hz * hz), 1 / 12.0 * mass * (hx * hx + hy * hy)));
			model_type = BOX;
			nObjects++;
			bData tData;
			tData.A = make_float3(pos->x, pos->y, pos->z);
			tData.B = make_float3(hx, hy, hz);
			tData.C = make_float3(0, 0, 0);
			tData.R = make_float4(rot->Get_A_quaternion().e0, rot->Get_A_quaternion().e1, rot->Get_A_quaternion().e2, rot->Get_A_quaternion().e3);
			tData.type = BOX;
			mData.push_back(tData);
			return true;
		}
		bool ChCollisionModelGPU::AddTriangle(ChVector<> A, ChVector<> B, ChVector<> C, ChVector<>* pos, ChMatrix33<>* rot) {			double mass = this->GetBody()->GetMass();
				model_type = TRIANGLEMESH;
			nObjects++;
			bData tData;
			tData.A = make_float3(A.x+pos->x, A.y+pos->y, A.z+pos->z);
			tData.B = make_float3(B.x+pos->x, B.y+pos->y, B.z+pos->z);
			tData.C = make_float3(C.x+pos->x, C.y+pos->y, C.z+pos->z);
			tData.R = make_float4(rot->Get_A_quaternion().e0, rot->Get_A_quaternion().e1, rot->Get_A_quaternion().e2, rot->Get_A_quaternion().e3);
			tData.type = TRIANGLEMESH;
			mData.push_back(tData);
			return true;
		}

		bool ChCollisionModelGPU::AddCylinder(double rx, double ry, double rz, ChVector<>* pos, ChMatrix33<>* rot) {
			double mass = this->GetBody()->GetMass();
			this->GetBody()->SetInertiaXX(ChVector<> (1 / 12.0 * mass * (3 * rx * rx + ry * ry), 1 / 2.0 * mass * (rx * rx), 1 / 12.0 * mass * (3 * rx * rx + ry * ry)));
			model_type = CYLINDER;
			nObjects++;
			bData tData;
			tData.A = make_float3(pos->x, pos->y, pos->z);
			tData.B = make_float3(rx, ry, rz);
			tData.C = make_float3(0, 0, 0);
			tData.R = make_float4(rot->Get_A_quaternion().e0, rot->Get_A_quaternion().e1, rot->Get_A_quaternion().e2, rot->Get_A_quaternion().e3);
			tData.type = CYLINDER;
			mData.push_back(tData);
			return true;
		}

		bool ChCollisionModelGPU::AddConvexHull(std::vector<ChVector<double> >& pointlist, ChVector<>* pos, ChMatrix33<>* rot) {
			//NOT SUPPORTED
			return false;
		}
		bool ChCollisionModelGPU::AddBarrel(double Y_low, double Y_high, double R_vert, double R_hor, double R_offset, ChVector<>* pos, ChMatrix33<>* rot) {
			//NOT SUPPORTED
			return false;
		}
		bool ChCollisionModelGPU::AddRectangle(double rx, double ry) {
			return false;
		}
		bool ChCollisionModelGPU::AddDisc(double rad) {
			return false;
		}
		bool ChCollisionModelGPU::AddEllipse(double rx, double ry) {
			return false;
		}
		bool ChCollisionModelGPU::AddCapsule(double len, double rad) {
			return false;
		}
		bool ChCollisionModelGPU::AddCone(double rad, double h) {
			return false;
		}
		/// Add a triangle mesh to this model
		bool ChCollisionModelGPU::AddTriangleMesh(const geometry::ChTriangleMesh& trimesh, bool is_static, bool is_convex, ChVector<>* pos, ChMatrix33<>* rot) {
			model_type = TRIANGLEMESH;
			nObjects += trimesh.getNumTriangles();
			bData tData;
			for (int i = 0; i < trimesh.getNumTriangles(); i++) {
				ChTriangle temptri = trimesh.getTriangle(i);
				tData.A = make_float3(temptri.p1.x+pos->x, temptri.p1.y+pos->y, temptri.p1.z+pos->z);
				tData.B = make_float3(temptri.p2.x+pos->x, temptri.p2.y+pos->y, temptri.p2.z+pos->z);
				tData.C = make_float3(temptri.p3.x+pos->x, temptri.p3.y+pos->y, temptri.p3.z+pos->z);
				tData.R = make_float4(rot->Get_A_quaternion().e0, rot->Get_A_quaternion().e1, rot->Get_A_quaternion().e2, rot->Get_A_quaternion().e3);
				tData.type = TRIANGLEMESH;
				mData.push_back(tData);
			}
			return true;
		}
		bool ChCollisionModelGPU::AddCopyOfAnotherModel(ChCollisionModel* another) {
			//NOT SUPPORTED
			return false;
		}
		void ChCollisionModelGPU::GetAABB(ChVector<>& bbmin, ChVector<>& bbmax) const {
		}

		void ChCollisionModelGPU::SetFamily(int mfamily) {
			colFam = mfamily;
		}

		int ChCollisionModelGPU::GetFamily() {
			return colFam;
		}

		void ChCollisionModelGPU::SetFamilyMaskNoCollisionWithFamily(int mfamily) {
			noCollWith = mfamily;
		}

		void ChCollisionModelGPU::SetFamilyMaskDoCollisionWithFamily(int mfamily) {
			if (noCollWith = mfamily) {
				noCollWith = -1;
			}
		}
		bool ChCollisionModelGPU::GetFamilyMaskDoesCollisionWithFamily(int mfamily) {
			return (noCollWith != mfamily);
		}

		int ChCollisionModelGPU::GetNoCollFamily() {
			return noCollWith;
		}
		void ChCollisionModelGPU::SyncPosition() {
			ChBody* bpointer = GetBody();
			assert(bpointer);
			//assert(bpointer->GetSystem());
		}

		ChPhysicsItem* ChCollisionModelGPU::GetPhysicsItem() {
			return (ChPhysicsItem*) GetBody();
		}
		;

	} // END_OF_NAMESPACE____
} // END_OF_NAMESPACE____

#endif  // end of ! CH_NOCUDA
