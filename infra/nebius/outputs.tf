output "instance_id" {
  value = nebius_compute_v1_instance.cuda_lab.id
}

output "public_ip" {
  value = nebius_vpc_v1_allocation.public_ip.status.details.allocated_cidr
}

output "ssh" {
  value = "ssh ubuntu@${trimprefix(nebius_vpc_v1_allocation.public_ip.status.details.allocated_cidr, "/32")}"
}
