

def remove(ll, val):
    if ll.front == None:
        return
    while ll.front.value == val:
        ll.front = ll.front.next
    cur_node = ll.front
    prev_node = ll.front
    while cur_node.next != None:
        if cur_node.value == val:
            prev_node.next = cur_node.next
            cur_node = cur_node.next
        else:
            prev_node = cur_node
            cur_node = cur_node.next
    return


